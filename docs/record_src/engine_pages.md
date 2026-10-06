| Weights | Runs (failed) | Identical text | Prompt (tokens/s) | Generation (tokens/s) | Generation against device copy | Load (ms) |
|---|---:|---:|---:|---:|---|---:|
| device copy (upstream) | 6 (0) | 6/6 | 817 | 24.03 | 1.000x [1.000, 1.000] | 943 |
| in place, ext4 page cache (4 KiB) | 6 (0) | 6/6 | 812 | 22.77 | 0.943x [0.918, 0.969] | 516 |
| in place, tmpfs with huge pages (2 MiB) | 6 (0) | 6/6 | 817 | 24.08 | 0.997x [0.971, 1.024] | 270 |
| in place, hugetlbfs (2 MiB) | 6 (0) | 6/6 | 817 | 24.04 | 0.996x [0.970, 1.023] | 267 |
