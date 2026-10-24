import os
import tempfile
import unittest

import h5py
import numpy as np

from bagofholding.h5.bag import H5Bag


class TestH5BagStorage(unittest.TestCase):
    """How H5Bag lays data out in the file: string encoding, compression, heap use."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "bag.h5")

    def tearDown(self):
        self.tmp.cleanup()

    def roundtrip(self, obj):
        H5Bag.save(obj, self.path)
        return H5Bag(self.path).load()

    def test_strings_roundtrip(self):
        # (strings with NUL characters were never supported: h5py refuses them in variable-length strings)
        for s in ["", "plain", "ünïcödé ✓", "x" * 10_000, "\ud800"]:
            with self.subTest(repr(s[:12])):
                self.assertEqual(s, self.roundtrip(s))

    def test_metadata_fields_are_fixed_length(self):
        H5Bag.save({"a": 1}, self.path)
        with h5py.File(self.path, "r") as f:
            attrs = dict(f["object"].attrs)
        self.assertTrue(attrs, msg="expected metadata attributes")
        for key, value in attrs.items():
            with self.subTest(key):
                self.assertIsInstance(
                    value, np.bytes_, msg="metadata should be fixed-length bytes"
                )

    def test_string_content_is_fixed_length(self):
        H5Bag.save("some text", self.path)
        with h5py.File(self.path, "r") as f:
            dtype = f["object"].dtype
        self.assertEqual("S", dtype.kind)

    def test_array_compression(self):
        big = np.linspace(0, 1, 1000)
        small = np.arange(3)
        H5Bag.save({"big": big, "small": small, "scalar": np.array(2.5)}, self.path)
        with h5py.File(self.path, "r") as f:
            datasets = {}
            f.visititems(
                lambda name, obj: (
                    datasets.__setitem__(name, obj.compression)
                    if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f"
                    else None
                )
            )
        self.assertIn("gzip", datasets.values(), msg="large arrays are compressed")
        loaded = H5Bag(self.path).load()
        np.testing.assert_array_equal(big, loaded["big"])
        np.testing.assert_array_equal(small, loaded["small"])
        self.assertEqual(loaded["scalar"], 2.5)

    def test_compression_can_be_disabled(self):
        class Plain(H5Bag):
            compression_threshold = None

        Plain.save(np.linspace(0, 1, 1000), self.path)
        with h5py.File(self.path, "r") as f:
            self.assertIsNone(f["object"].compression)

    def test_empty_and_bytes_arrays(self):
        for arr in [
            np.zeros((0, 3)),
            np.array([b"a", b"abc"], dtype="S3"),
            np.array([b"x" * 40] * 100, dtype="S40"),
        ]:
            with self.subTest(arr.dtype.str):
                out = self.roundtrip(arr)
                self.assertEqual(arr.dtype, out.dtype)
                np.testing.assert_array_equal(arr, out)

    def test_repeated_saves_do_not_leak_heap(self):
        # Each save is its own write session; variable-length strings used to leave a
        # fresh >= 4 kB global heap collection behind every time.
        sizes = []
        for i in range(40):
            H5Bag.save({"name": f"entry {i}", "value": i}, f"{self.path}/e{i}")
            sizes.append(os.path.getsize(self.path))
        per_save = (sizes[-1] - sizes[0]) / (len(sizes) - 1)
        self.assertLess(per_save, 4096, msg=f"{per_save:.0f} bytes per save")


if __name__ == "__main__":
    unittest.main()
