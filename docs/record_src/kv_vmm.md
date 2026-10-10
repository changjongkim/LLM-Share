| Parent's instance | Prefix (tokens) | Hand-over | Runs | Pause of the parent (ms) | State file (MiB) | Children that ran | Child texts equal to copy | Child attach (ms) | Child first token (ms) | Child generation (tokens/s, each) | Parent generation (tokens/s) | Memory of the children (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12-SM | 4081 | copy (state file with the rows) | 3 | 216.2 | 223.24 | 12/12 | 12/12 | 192 | 1581 | 5.34 | 7.63 | 4546 |
| 12-SM | 4081 | host memory: mapped file, private tail | 3 | 3.0 | 0.06 | 12/12 | 12/12 | 27 | 1363 | 5.25 | 6.89 | 2948 |
| 12-SM | 4081 | device memory: shared allocations, private tail | 3 | 7.6 | 0.06 | 12/12 | 12/12 | 325 | 1684 | 5.41 | 7.53 | 3807 |
| 12-SM | 4081 | host memory, one child in the other instance | 3 | 2.8 | 0.06 | 3/3 | 3/3 | 25 | 701 | 15.58 | 23.88 | 809 |
| 12-SM | 4081 | device memory, one child in the other instance | 3 | 7.3 | 0.06 | 0/3 | - | - | - | - | 24.70 | 386 |
| 8-SM | 4081 | copy (state file with the rows) | 3 | 213.6 | 223.24 | 12/12 | 12/12 | 208 | 1678 | 3.28 | 3.98 | 4553 |
| 8-SM | 4081 | host memory: mapped file, private tail | 3 | 3.0 | 0.06 | 12/12 | 12/12 | 27 | 1479 | 3.16 | 3.68 | 2876 |
| 8-SM | 4081 | device memory: shared allocations, private tail | 3 | 7.2 | 0.06 | 12/12 | 12/12 | 332 | 1762 | 3.31 | 4.03 | 3727 |
| 8-SM | 4081 | host memory, one child in the other instance | 3 | 2.8 | 0.06 | 3/3 | 3/3 | 24 | 693 | 23.69 | 15.49 | 837 |
| 8-SM | 4081 | device memory, one child in the other instance | 3 | 7.4 | 0.06 | 0/3 | - | - | - | - | 16.16 | 416 |
| 12-SM | 16321 | copy (state file with the rows) | 3 | 551.1 | 892.80 | 12/12 | 12/12 | 500 | 1882 | 5.11 | 7.50 | 10148 |
| 12-SM | 16321 | host memory: mapped file, private tail | 3 | 8.2 | 0.25 | 12/12 | 12/12 | 48 | 1432 | 5.00 | 6.54 | 3078 |
| 12-SM | 16321 | device memory: shared allocations, private tail | 3 | 41.0 | 0.25 | 12/12 | 12/12 | 438 | 1835 | 5.14 | 7.48 | 3968 |
| 12-SM | 16321 | host memory, one child in the other instance | 3 | 7.9 | 0.25 | 3/3 | 0/3 | 44 | 728 | 14.48 | 22.08 | 835 |
| 12-SM | 16321 | device memory, one child in the other instance | 3 | 42.6 | 0.25 | 0/3 | - | - | - | - | 23.37 | 271 |
| 8-SM | 16321 | copy (state file with the rows) | 3 | 570.6 | 892.80 | 12/12 | 12/12 | 524 | 2000 | 3.24 | 4.01 | 10092 |
| 8-SM | 16321 | host memory: mapped file, private tail | 3 | 9.0 | 0.25 | 12/12 | 12/12 | 49 | 1553 | 2.98 | 3.44 | 3040 |
| 8-SM | 16321 | device memory: shared allocations, private tail | 3 | 39.1 | 0.25 | 12/12 | 12/12 | 434 | 1915 | 3.25 | 4.02 | 3904 |
| 8-SM | 16321 | host memory, one child in the other instance | 3 | 8.0 | 0.25 | 3/3 | 3/3 | 43 | 721 | 21.71 | 14.32 | 829 |
| 8-SM | 16321 | device memory, one child in the other instance | 3 | 39.5 | 0.25 | 0/3 | - | - | - | - | 15.66 | 305 |
