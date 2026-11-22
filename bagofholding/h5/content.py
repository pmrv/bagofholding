# Designed and intended for use with H5 implementations, which treat arrays very nicely
# but since bags merely follow a protocol, there is nothing intrinsically-h5 here
from __future__ import annotations

import struct
from collections.abc import Hashable
from typing import Any, Literal, Protocol, TypeAlias, cast

import numpy as np

from bagofholding.content import (
    BespokeItem,
    Item,
    Packer,
    PackingArguments,
    UnpackingArguments,
    has_surrogates,
    pack,
    unpack,
)
from bagofholding.h5.dtypes import H5DtypeAlias

_INT64_MIN = -9_223_372_036_854_775_808
_UINT64_MAX = 18_446_744_073_709_551_615


def int_overflows(n: int) -> bool:
    """h5py breaks on very extreme integers. Catch them."""
    return n < _INT64_MIN or n > _UINT64_MAX


ArrayType: TypeAlias = np.ndarray[tuple[int, ...], H5DtypeAlias]


class ArrayPacker(Packer, Protocol):
    def pack_array(self, obj: ArrayType, path: str) -> None: ...
    def unpack_array(self, path: str) -> ArrayType: ...


class Array(BespokeItem[ArrayType, ArrayPacker]):
    @classmethod
    def _pack_item(cls, obj: ArrayType, packer: ArrayPacker, path: str) -> None:
        packer.pack_array(obj, path)

    @classmethod
    def unpack(
        cls, packer: ArrayPacker, path: str, unpacking: UnpackingArguments
    ) -> ArrayType:
        return packer.unpack_array(path)


ObjectArrayType: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.object_]]


class StringArrayPacker(ArrayPacker, Protocol):
    def pack_string_array(self, obj: list[str], path: str) -> None: ...
    def unpack_string_array(self, path: str) -> list[str]: ...


def is_plain_object_array(obj: object) -> bool:
    return (
        type(obj) is np.ndarray
        and obj.dtype == np.dtype(object)
        and obj.dtype.metadata is None
    )


def _string_array_compatible(s: str) -> bool:
    # HDF5 variable-length strings are UTF-8 and NUL-terminated
    return "\x00" not in s and not has_surrogates(s)


def _dedup_key(element: object) -> Hashable:
    """
    Plain immutable scalars are deduplicated by value -- bags never preserve their
    identity anyway. Everything else is deduplicated by identity, so references
    between objects survive exactly as they would through the regular packing path.
    """
    t = type(element)
    if t is float:
        # Compare bitwise to keep -0.0 and NaN payloads distinct
        return (t, struct.pack("<d", element))
    if t in (str, int, bool, bytes):
        return (t, element)
    return id(element)


class Strings(Item[list[str], list[str], StringArrayPacker]):
    """A flat list of strings, stored as a single array."""

    @classmethod
    def _pack_item(cls, obj: list[str], packer: StringArrayPacker, path: str) -> None:
        packer.pack_string_array(obj, path)

    @classmethod
    def unpack(
        cls, packer: StringArrayPacker, path: str, unpacking: UnpackingArguments
    ) -> list[str]:
        return packer.unpack_string_array(path)


class ObjectArray(BespokeItem[ObjectArrayType, StringArrayPacker]):
    """
    Numpy arrays of :class:`object` dtype, e.g. the backing store of pandas columns
    holding strings, enums, or other python objects.

    Reducing these the way pickle does produces one stored node per element, which
    makes long arrays very slow to save and load. Instead, elements are deduplicated
    and stored once each, along with an integer array indexing into these unique
    values. Unique plain strings are stored together as a single string array; any
    other unique elements are packed as a single tuple, following the regular rules
    (and in particular the regular memoization of shared references).
    """

    @classmethod
    def pack(
        cls,
        obj: ObjectArrayType,
        packer: StringArrayPacker,
        path: str,
        packing: PackingArguments,
    ) -> None:
        fortran = bool(obj.flags.f_contiguous and not obj.flags.c_contiguous)
        flat = obj.ravel(order="F" if fortran else "C")

        # Unique elements are numbered in order of first appearance, then the stored
        # index is remapped to point into the concatenation [*strings, *others]
        first_seen: dict[Hashable, int] = {}
        strings: list[str] = []
        others: list[object] = []
        is_string: list[bool] = []
        seen_order = np.empty(flat.size, dtype=np.int64)
        for i, element in enumerate(flat):
            key = _dedup_key(element)
            n = first_seen.get(key)
            if n is None:
                n = first_seen[key] = len(is_string)
                if type(element) is str and _string_array_compatible(element):
                    strings.append(element)
                    is_string.append(True)
                else:
                    others.append(element)
                    is_string.append(False)
            seen_order[i] = n
        string_mask = np.array(is_string, dtype=bool)
        remap = np.empty(len(is_string), dtype=np.int64)
        remap[string_mask] = np.arange(len(strings))
        remap[~string_mask] = len(strings) + np.arange(len(others))
        index = remap[seen_order]

        packer.create_group(path)
        packer.pack_metadata(cls._get_metadata(obj, packing), path)
        for subpath, value in (
            ("shape", np.array(obj.shape, dtype=np.int64)),
            ("fortran", fortran),
            ("index", index),
            ("others", tuple(others)),
        ):
            pack(
                value,
                packer,
                packer.join(path, subpath),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
            )
        Strings.pack(strings, packer, packer.join(path, "strings"), packing)

    @classmethod
    def _pack_item(
        cls, obj: ObjectArrayType, packer: StringArrayPacker, path: str
    ) -> None:
        raise NotImplementedError(f"{cls.__name__} overrides `pack` directly")

    @classmethod
    def unpack(
        cls, packer: StringArrayPacker, path: str, unpacking: UnpackingArguments
    ) -> ObjectArrayType:
        def sub(name: str) -> Any:
            return unpack(
                packer,
                packer.join(path, name),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )

        shape = tuple(int(n) for n in sub("shape"))
        order: Literal["C", "F"] = "F" if sub("fortran") else "C"
        obj: ObjectArrayType = np.empty(shape, dtype=object, order=order)
        # Memoize before unpacking elements, which may reference this very array
        unpacking.memo[path] = obj

        strings = cast(list[str], sub("strings"))
        others = cast(tuple[object, ...], sub("others"))
        uniques = np.empty(len(strings) + len(others), dtype=object)
        # Assign one at a time: bulk assignment would broadcast sequence elements
        for i, element in enumerate((*strings, *others)):
            uniques[i] = element
        obj.ravel(order=order)[:] = uniques[sub("index")]
        return obj
