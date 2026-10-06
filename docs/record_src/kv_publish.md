| Prefix (tokens) | Way | Runs (failed) | Compute the prefix (ms) | Hand it over (ms) | State file (MiB) | Cache file in use (MiB) |
|---:|---|---:|---:|---:|---:|---:|
| 4081 | save the state file (upstream) | 6 (0) | 5405 | 230.8 | 223.24 | - |
| 4081 | publish, cache file with 2 MiB pages | 6 (0) | 5443 | 3.0 | 0.06 | 224 |
| 4081 | publish, cache file with 4 KiB pages | 6 (0) | 5556 | 4.3 | 0.06 | 224 |
| 16321 | save the state file (upstream) | 6 (0) | 24453 | 561.2 | 892.80 | - |
| 16321 | publish, cache file with 2 MiB pages | 6 (0) | 24646 | 9.1 | 0.25 | 896 |
| 16321 | publish, cache file with 4 KiB pages | 6 (0) | 24816 | 14.2 | 0.25 | 896 |
