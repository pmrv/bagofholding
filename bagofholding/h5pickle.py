"""
Proof of concept: `bagofholding` as a :class:`pickle.Pickler`/:class:`pickle.Unpickler`.

`pickle` walks the object. We only

1. name every `save` call with the path `bagofholding` would give that object,
2. remember which bytes of the pickle stream that call produced, and
3. push leaf data out-of-band (via `persistent_id`) into datasets at those paths.

Each `save` call emits a self-contained stack program, so any path can be reloaded
by running just its byte range, plus the ranges that define the memo entries it
refers back to (shared references, and enclosing objects in case of cycles).

File layout::

    /pickle/stream   the flat pickle byte stream
    /pickle/paths    the path of each node
    /pickle/nodes    per path: (start, end, memo index or -1) into the stream
    /pickle/memo     per memo index: (start, end) of the `save` call defining it
    /pickle/gets     per memo lookup in the stream: (position, memo index)
    /object/...      out-of-band data, at the same paths as listed in /pickle/paths
"""

# typeshed does not describe the internals of the pure-python pickler we build on
# mypy: disable-error-code="misc, attr-defined, no-any-return"

from __future__ import annotations

import functools
import io
import itertools
import operator
import pathlib
import pickle
import posixpath
import struct
import types
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np

from bagofholding.h5.dtypes import H5PY_DTYPE_WHITELIST

ROOT = "object"
PROTOCOL = 4  # like `bagofholding`, so that objects reduce to the very same parts
LOADERS: dict[str, Callable[[Any], Any]] = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "str": lambda raw: raw.decode("utf-8"),
    "bytes": lambda raw: raw.tobytes(),
    "bytearray": bytearray,
    "ndarray": np.asarray,
}
SCALARS = (bool, int, float, complex, str, bytes)
SAME = ""  # label for a child that takes the path of its parent
VIRTUAL = (-1, -1, -1)  # node of a container that only `bagofholding` has, e.g. `keys`
VIRTUAL_LISTS = ("item_iterator", "kv_iterator/args/i0")  # the others are tuples


def _union(*members: Any) -> Any:
    return functools.reduce(operator.or_, members)


class _Raw:
    """
    Hides `copyreg.__newobj__` from `save_reduce`, which would otherwise fold the
    class into a NEWOBJ opcode instead of saving `constructor` and `args` as they are.
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func

    def __call__(self) -> None: ...


class _Frame:
    """One `save` call: its path, and how to name the `save` calls it makes."""

    def __init__(self, path: str | None, obj: object = None) -> None:
        t = type(obj)
        self.path = path
        self.labels: list[str | None] = []  # for the parts of a reduction
        self.mode = (
            "seq" if t in (tuple, list, set, frozenset) else "kv" if t is dict else None
        )
        self.prefix = ""
        self.str_keys = isinstance(obj, dict) and all(
            type(k) is str and k and k.isascii() and "/" not in k for k in obj
        )
        self.n, self.last = 0, None
        self.index = -1  # of this object in the pickle memo, if it got or used one

    def label(self, child: Any) -> str | None:
        (i, odd), key = divmod(self.n, 2), self.last
        self.n, self.last = self.n + 1, child
        if self.mode == "seq":
            return f"{self.prefix}i{self.n - 1}"
        if self.mode == "kv":  # pickle saves key, value, key, value, ...
            if self.prefix:
                return f"{self.prefix}i{i}/i{odd}"
            if self.str_keys:
                return key if odd else None
            return f"{'values' if odd else 'keys'}/i{i}"
        return self.labels.pop(0) if self.labels else None


class H5Pickler(pickle._Pickler):  # the C pickler has no hook to learn nesting from
    def __init__(self, h5: h5py.Group, scalars: bool = True) -> None:
        """
        Args:
            h5: Where to write.
            scalars: Whether builtin scalars go out-of-band like arrays do (except
                inside sequences), or stay in the pickle stream.
        """
        self._stream = io.BytesIO()
        super().__init__(self._stream, protocol=PROTOCOL)
        self.h5, self.scalars = h5, scalars
        self._frames = [_Frame("")]
        self._frames[0].labels = [ROOT]
        self._nodes: dict[str, tuple[int, int, int]] = {}
        self._memo_spans: dict[int, tuple[int, int]] = {}
        self._gets: list[tuple[int, int]] = []
        self._stored: dict[int, tuple[str, Any]] = {}

    def dump(self, obj: Any) -> None:
        self.save(obj)
        everything = (0, self._stream.tell())
        g = self.h5.create_group("pickle")
        g["stream"] = np.frombuffer(self._stream.getvalue(), np.uint8)
        g.create_dataset("paths", data=list(self._nodes), dtype=h5py.string_dtype())
        g["nodes"] = np.array(list(self._nodes.values()), np.int64)
        g["memo"] = np.array(
            [self._memo_spans.get(i, everything) for i in range(len(self.memo))],
            np.int64,
        ).reshape(-1, 2)
        g["gets"] = np.array(self._gets, np.int64).reshape(-1, 2)

    def save(self, obj: Any, save_persistent_id: bool = True) -> None:
        if type(obj) is _Raw:
            obj = obj.func
        parent = self._frames[-1]
        label = parent.label(obj)
        if parent.path is None or label is None:
            path = None
        elif label == SAME:
            path = parent.path
        else:
            path = posixpath.join(parent.path, label)
            parts = label.split("/")
            for i in range(1, len(parts)):
                if parts[i][1:].isdigit():  # an item, so what holds it is a container
                    container = posixpath.join(parent.path, *parts[:i])
                    self._nodes.setdefault(container, VIRTUAL)
        frame = _Frame(path, obj)
        self._frames.append(frame)
        start, n_memo = self._stream.tell(), len(self.memo)
        try:
            super().save(obj, save_persistent_id)
        finally:
            self._frames.pop()
        span = (start, self._stream.tell())
        index = self.memo.get(id(obj), (-1,))[0]
        if index >= n_memo:  # memoized by this very call
            self._memo_spans.setdefault(index, span)
            frame.index = index
        if path is not None:
            self._nodes[path] = (*span, frame.index)

    def persistent_id(self, obj: Any) -> str | None:
        path, t = self._frames[-1].path, type(obj)
        shared = t is bytearray or (
            t is np.ndarray and obj.dtype.type in H5PY_DTYPE_WHITELIST
        )
        if shared and id(obj) in self._stored:  # mutable, so stored once per object
            return f"{t.__name__}:{self._stored[id(obj)][0]}"
        named = self.scalars and t in SCALARS and self._frames[-2].mode != "seq"
        if path is None or not (shared or named) or path in self.h5:
            return None
        try:
            self.h5.create_dataset(path, data=np.void(obj) if t is bytes else obj)
        except (TypeError, ValueError, OverflowError):
            # No HDF5 equivalent, so it just stays in the pickle stream
            if path in self.h5:
                del self.h5[path]
            return None
        if shared:
            self._stored[id(obj)] = (path, obj)
        return f"{t.__name__}:{path}"

    def reducer_override(self, obj: Any) -> Any:
        if type(obj) is types.UnionType:  # `bagofholding` stores just the members
            return _union, obj.__args__
        return NotImplemented

    def put(self, idx: int) -> bytes:
        # Explicit memo indices (instead of MEMOIZE), so byte ranges run in isolation
        return pickle.LONG_BINPUT + struct.pack("<I", idx)

    def get(self, i: int) -> bytes:
        self._gets.append((self._stream.tell(), i))
        self._frames[-1].index = i
        return super().get(i)

    def save_reduce(
        self,
        func: Any,
        args: Any,
        state: Any = None,
        listitems: Any = None,
        dictitems: Any = None,
        state_setter: Any = None,
        *,
        obj: Any = None,
    ) -> None:
        if func is _union:
            self._frames[-1].labels = [None, SAME]
        elif not isinstance(obj, type):  # e.g. NoneType is a leaf to `bagofholding`
            labels: list[str | None] = ["constructor", "args"]
            if state is not None:
                labels += (
                    ["state"] if state_setter is None else ["setter", None, "state"]
                )
            self._frames[-1].labels = labels
        if getattr(func, "__name__", "") in ("__newobj__", "__newobj_ex__"):
            func = _Raw(func)
        super().save_reduce(
            func, args, state, listitems, dictitems, state_setter, obj=obj
        )

    def _batch_appends(self, *args: Any) -> None:
        self._batch(super()._batch_appends, "seq", "item_iterator/", *args)

    def _batch_setitems(self, *args: Any) -> None:
        self._batch(super()._batch_setitems, "kv", "kv_iterator/args/i0/", *args)

    def _batch(
        self, batch: Callable[..., None], mode: str, prefix: str, *args: Any
    ) -> None:
        frame = self._frames[-1]
        if frame.mode is None:  # items of a reduction, not of a plain list or dict
            frame.mode, frame.prefix, frame.n = mode, prefix, 0
            batch(*args)
            frame.mode = None
        else:
            batch(*args)


class H5Unpickler(pickle.Unpickler):
    def __init__(self, h5: h5py.Group, path: str = ROOT) -> None:
        self.h5 = h5
        self._loaded: dict[str, Any] = {}
        g = h5["pickle"]
        nodes = dict(zip(g["paths"].asstr()[()], g["nodes"][()].tolist(), strict=True))
        stream: bytes = g["stream"][()].tobytes()
        memo, gets = g["memo"][()], g["gets"][()]
        todo: list[tuple[int, int]] = []

        def push(path: str) -> bytes:
            """A program for the object at `path`, given the memo entries in `todo`."""
            start, end, index = nodes[path]
            if start < 0:
                names = (f"{path}/i{i}" for i in itertools.count())
                items = itertools.takewhile(nodes.__contains__, names)
                close = pickle.LIST if path.endswith(VIRTUAL_LISTS) else pickle.TUPLE
                return b"".join([pickle.MARK, *map(push, items), close])
            if index < 0:  # not memoized, so it refers to nothing either
                return stream[start:end]
            todo.append((start, end))  # may sit inside a wider range, due to cycles
            return pickle.LONG_BINGET + struct.pack("<I", index)

        target = push(path)
        spans = set()
        while todo:
            span = todo.pop()
            if span not in spans:
                spans.add(span)
                # Memo entries looked up in this range, but defined before it
                lo, hi = np.searchsorted(gets[:, 0], span)
                needs = memo[gets[lo:hi, 1]]
                needs = np.unique(needs[needs[:, 0] < span[0]], axis=0)
                todo += map(tuple, needs.tolist())
        program, done = [pickle.PROTO, bytes([PROTOCOL]), pickle.MARK], 0
        for start, end in sorted(spans, key=lambda span: (span[0], -span[1])):
            if start >= done:  # else it is part of the previous range
                program.append(stream[start:end])
                done = end
        program += [target, pickle.TUPLE, pickle.STOP]
        super().__init__(io.BytesIO(b"".join(program)))

    def load(self) -> Any:
        return super().load()[-1]

    def persistent_load(self, pid: str) -> Any:
        kind, _, path = pid.partition(":")
        if path not in self._loaded:
            self._loaded[path] = LOADERS[kind](self.h5[path][()])
        return self._loaded[path]


def dump(obj: Any, filepath: str | pathlib.Path, scalars: bool = True) -> None:
    with h5py.File(filepath, "w") as f:
        H5Pickler(f, scalars=scalars).dump(obj)


def load(filepath: str | pathlib.Path, path: str = ROOT) -> Any:
    with h5py.File(filepath, "r") as f:
        return H5Unpickler(f, path).load()


def list_paths(filepath: str | pathlib.Path) -> list[str]:
    with h5py.File(filepath, "r") as f:
        return sorted(f["pickle/paths"].asstr()[()])
