| Weights | Sequences per server | Runs (failed) | Generation, both servers (tokens/s) | 12-SM instance | 6-SM instance | Memory outside the model file (MiB) | Model file mapped (MiB) | Total (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| device copy | 4 | 6 (0) | 126.0 +/- 0.1 | 75.0 | 51.1 | 11366 | 0 | 11366 |
| in place | 4 | 6 (0) | 123.5 +/- 0.2 | 73.2 | 50.2 | 2324 | 4460 | 6784 |
| device copy | 8 | 6 (0) | 149.5 +/- 0.1 | 89.2 | 60.4 | 11362 | 0 | 11362 |
| in place | 8 | 6 (0) | 149.0 +/- 0.2 | 88.9 | 60.1 | 2324 | 4460 | 6784 |
