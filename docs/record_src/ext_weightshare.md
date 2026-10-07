| Placement | Stack | Runs | Agents complete | In the other instance | Library: workers / fallbacks | Weights load (ms) | First token (ms) | Throughput (tokens/s) | Memory (MiB) | 95% CI | Texts equal to stock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same | stock | 6 | 48/48 | 0/0 | 0 / 0 | 2616 | 3824 | 24.33 | 55206 | 164 | 48/48 |
| same | ipc | 6 | 48/48 | 0/0 | 48 / 0 | 2020 | 3240 | 22.90 | 21585 | 71 | 48/48 |
| same | inplace | 6 | 48/48 | 0/0 | 0 / 0 | 1138 | 2383 | 22.85 | 14614 | 38 | 48/48 |
| cross | stock | 6 | 48/48 | 24/24 | 0 / 0 | 2540 | 3527 | 37.88 | 55162 | 31 | 48/48 |
| cross | ipc | 6 | 24/48 | 0/24 | 24 / 24 | 2374 | 3210 | 22.31 | 38267 | 46 | 24/24 |
| cross | inplace | 6 | 48/48 | 24/24 | 0 / 0 | 1119 | 2161 | 34.87 | 14523 | 77 | 48/48 |
