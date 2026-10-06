| Weights | Runs (failed) | Identical text | Prompt (tokens/s) | Generation (tokens/s) | Generation against device copy | Load (ms) | Memory outside the model file (MiB) | Model file mapped (MiB) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| device copy (upstream) | 6 (0) | 6/6 | 817 | 24.12 | 1.000x [1.000, 1.000] | 942 | 5423 | 0 |
| in place, CPU reads every page at load | 6 (0) | 6/6 | 813 | 22.79 | 0.940x [0.914, 0.967] | 517 | 878 | 4460 |
| in place, no CPU read | 6 (0) | 6/6 | 812 | 22.80 | 0.940x [0.912, 0.969] | 1013 | 921 | 4460 |
