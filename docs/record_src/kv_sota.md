| Placement | Hand-over | Children complete | Texts equal to copy | Parent pause (ms) | Attach (ms) | First token (ms) | Memory (MiB) | Throughput (tokens/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| same | copy | 48/48 | 48/48 | 539.54 | 548.6 | 2670 | 20306 | 21.85 |
| same | demand-backed copy | 48/48 | 48/48 | 542.80 | 596.6 | 2721 | 14210 | 21.98 |
| same | shared device memory | 48/48 | 48/48 | 41.04 | 769.5 | 2864 | 7808 | 22.33 |
| same | copy-on-write | 48/48 | 48/48 | 8.12 | 50.3 | 2770 | 6892 | 21.54 |
| same | extents | 48/48 | 48/48 | 8.37 | 48.8 | 2137 | 6036 | 21.74 |
| cross | copy | 48/48 | 48/48 | 548.04 | 494.8 | 2395 | 20259 | 33.23 |
| cross | demand-backed copy | 48/48 | 48/48 | 539.12 | 559.1 | 2445 | 13732 | 33.55 |
| cross | shared device memory | 24/48 | 24/24 | 39.73 | 648.6 | 2541 | 3984 | 21.73 |
| cross | copy-on-write | 48/48 | 48/48 | 8.87 | 50.9 | 2279 | 6785 | 32.05 |
| cross | extents | 48/48 | 48/48 | 8.00 | 50.4 | 1918 | 5923 | 32.29 |
