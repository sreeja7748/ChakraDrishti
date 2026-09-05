"""
inspect_h5.py
--------------
Run this FIRST against any downloaded TSRD .h5 file, before using
tsrd_loader.py. It prints the file's actual internal structure -
every group and dataset, with shapes/dtypes and a small data preview -
so we can see the REAL schema instead of guessing from the README.

Usage:
    pip install h5py --break-system-packages
    python inspect_h5.py path/to/config_0.h5
"""

import sys
import h5py
import numpy as np


def describe(name, obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, h5py.Group):
        print(f"{pad}[GROUP] {name}/")
        for key in obj.keys():
            describe(key, obj[key], indent + 1)
    elif isinstance(obj, h5py.Dataset):
        print(f"{pad}[DATASET] {name}  shape={obj.shape}  dtype={obj.dtype}")
        try:
            preview = obj[()]
            if isinstance(preview, np.ndarray) and preview.size > 0:
                flat = preview.reshape(-1)
                sample = flat[:5]
                print(f"{pad}    first values: {sample}")
        except Exception as e:
            print(f"{pad}    (couldn't preview: {e})")
        # datasets can carry their own attributes too (units, column names, etc.)
        if obj.attrs:
            print(f"{pad}    attrs: {dict(obj.attrs)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py path/to/config_0.h5")
        return

    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        print(f"=== Top-level structure of {path} ===\n")
        # file-level attributes (often metadata like emitter count, duration, etc.)
        if f.attrs:
            print("File-level attrs:", dict(f.attrs), "\n")
        for key in f.keys():
            describe(key, f[key])


if __name__ == "__main__":
    main()