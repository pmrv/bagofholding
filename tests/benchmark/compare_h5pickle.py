"""
Compare the `bagofholding.h5pickle` proof of concept against the existing bags.

Run as `python tests/benchmark/compare_h5pickle.py`.
"""

import collections
import os
import pathlib
import pickle
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from static.objects import build_cases  # noqa: E402

import bagofholding as boh  # noqa: E402
from bagofholding import h5pickle  # noqa: E402


def same(a, b):
    if type(a) is not type(b):
        return False
    try:
        if pickle.dumps(a, 4) == pickle.dumps(b, 4):
            return True
    except Exception:
        pass
    try:
        return bool(np.all(a == b))
    except Exception:
        return False


def attributes(root):
    """The attribute names of every instance reachable from root."""
    seen, todo, found = set(), [root], []
    while todo:
        obj = todo.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            todo += [*obj.keys(), *obj.values()]
        elif isinstance(obj, (list, tuple, set, frozenset)):
            todo += obj
        elif hasattr(obj, "__self__"):
            todo.append(obj.__self__)
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            found.append(f"{type(obj).__name__}{sorted(vars(obj))}")
            todo += vars(obj).values()
    return ", ".join(sorted(found))


def compare_paths(tmp):
    """Every path of the original must partially load to the same thing here."""
    n_orig = n_poc = n_common = 0
    missing = collections.Counter()
    problems = []
    for name, obj in build_cases().items():
        orig, poc = f"{tmp}/{name}_orig.h5", f"{tmp}/{name}_poc.h5"
        boh.H5Bag.save(obj, orig)
        h5pickle.dump(obj, poc)
        if not same(obj, h5pickle.load(poc)):
            problems.append(f"{name}: full round trip differs")
        bag = boh.H5Bag(orig)
        orig_paths, poc_paths = set(bag.list_paths()), set(h5pickle.list_paths(poc))
        n_orig, n_poc = n_orig + len(orig_paths), n_poc + len(poc_paths)
        for path in sorted(orig_paths - poc_paths):
            kind = bag[path].content_type.rsplit(".", 1)[-1]
            missing[f"{path.rsplit('/', 1)[-1].rstrip('0123456789')} ({kind})"] += 1
        problems += [f"{name}: extra path {p}" for p in poc_paths - orig_paths]
        for path in sorted(orig_paths & poc_paths):
            n_common += 1
            a, b = bag.load(path), h5pickle.load(poc, path)
            if not same(a, b):
                problems.append(
                    f"{name}: partial load differs at {path}\n"
                    f"      original holds {attributes(a)}\n"
                    f"      poc holds      {attributes(b)}"
                )
    print(f"paths: original {n_orig}, poc {n_poc}, partial loads compared {n_common}")
    print("original paths without a poc counterpart, by last path element (content):")
    for what, n in missing.most_common():
        print(f"  {n:4d} x {what}")
    print("problems:", *(problems or ["none"]), sep="\n  ")


class Leaf:
    def __init__(self, i):
        self.i, self.x, self.label = i, float(i), f"leaf{i}"
        self.data = np.arange(i, i + 10.0)


WORKLOADS = {
    "2000 small objects": (
        {f"k{i}": Leaf(i) for i in range(2000)},
        "object/k1999/state/data",
    ),
    "20 arrays of 1e6 floats": (
        {f"k{i}": np.random.rand(1_000_000) for i in range(20)},
        "object/k19",
    ),
    "list of 20000 floats": (
        {"values": np.random.rand(20_000).tolist()},
        "object/values/i19999",
    ),
}
BACKENDS = {
    "H5Bag": (boh.H5Bag.save, lambda f, p="object": boh.H5Bag(f).load(p)),
    "TrieH5Bag": (boh.TrieH5Bag.save, lambda f, p="object": boh.TrieH5Bag(f).load(p)),
    "h5pickle": (h5pickle.dump, h5pickle.load),
    "h5pickle, scalars in stream": (
        lambda o, f: h5pickle.dump(o, f, scalars=False),
        h5pickle.load,
    ),
}


def timed(func, *args):
    t0 = time.perf_counter()
    func(*args)
    return time.perf_counter() - t0


def compare_speed(tmp):
    for workload, (obj, leaf) in WORKLOADS.items():
        print(f"\n{workload} (partial load: {leaf})")
        print(f"  {'':28s}{'save':>9s}{'load':>9s}{'partial':>10s}{'size':>10s}")
        for backend, (save, load) in BACKENDS.items():
            file = f"{tmp}/speed.h5"
            t_save, t_load, t_part = (
                timed(save, obj, file),
                timed(load, file),
                timed(load, file, leaf),
            )
            size = os.path.getsize(file) / 1e6
            print(
                f"  {backend:28s}{t_save:8.2f}s{t_load:8.2f}s{t_part * 1e3:8.1f}ms"
                f"{size:8.1f}MB"
            )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        compare_paths(tmp)
        compare_speed(tmp)
