#!/usr/bin/env python3
"""Compares two cache files of a publisher byte by byte.

usage: kv_file_diff.py FILE_A FILE_B TENSOR_BYTES

Prints the number of bytes that differ, the number of 16-bit values that
differ, the largest difference between two such values read as half floats,
and the index of the first tensor (TENSOR_BYTES each, in the order the engine
lays them out: key and value tensor of layer 0, of layer 1, ...) that holds a
difference.
"""
import sys

import numpy

path_a, path_b, tensor_bytes = sys.argv[1], sys.argv[2], int(sys.argv[3])
a = numpy.memmap(path_a, dtype=numpy.uint8, mode="r")
b = numpy.memmap(path_b, dtype=numpy.uint8, mode="r")
if a.size != b.size:
    sys.exit("the files differ in size")
bytes_differ = 0
values_differ = 0
largest = 0.0
first = -1
chunk = 64 << 20
for start in range(0, a.size, chunk):
    x = numpy.asarray(a[start:start + chunk])
    y = numpy.asarray(b[start:start + chunk])
    unequal = x != y
    count = int(unequal.sum())
    if count == 0:
        continue
    if first < 0:
        first = (start + int(unequal.argmax())) // tensor_bytes
    bytes_differ += count
    hx = x.view(numpy.float16).astype(numpy.float32)
    hy = y.view(numpy.float16).astype(numpy.float32)
    values = hx != hy
    values_differ += int(values.sum())
    delta = numpy.abs(hx - hy)
    largest = max(largest, float(delta[numpy.isfinite(delta)].max(initial=0.0)))
print(f"bytes_differ={bytes_differ} values_differ={values_differ} "
      f"largest_difference={largest:.6g} first_tensor={first}")
