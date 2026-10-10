| Configuration | Runs (failed servers) | All agents ready (ms) | Throughput (tokens/s) | 12-SM | 8-SM | Second server attach (ms) | Publish (ms) | State file (MiB) | Memory (MiB) | Texts equal to copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one_server | 6 (0) | 20775 | 75.71 | 75.71 | 0.00 | 0.0 | 0.00 | 0.00 | 2584 | 36/48 |
| two_servers_compute | 6 (0) | 30693 | 105.74 | 61.27 | 44.47 | 82.8 | 0.00 | 0.00 | 5062 | 36/48 |
| two_servers_copy | 6 (0) | 22168 | 104.48 | 61.03 | 43.45 | 245.0 | 560.32 | 892.80 | 5086 | 48/48 |
| two_servers_extent | 6 (0) | 21484 | 102.28 | 60.51 | 41.76 | 43.8 | 11.34 | 0.25 | 2448 | 48/48 |
