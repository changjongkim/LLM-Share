| Processes | Mode | Runs (failed) | Pages faulted in for the GPU | lowest | highest | First token (ms) | Throughput (tokens/s) | Texts equal to copy | Pages before the publish | Publish (ms) | Next task (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agents | copy | 6 (0) | 0 | 0 | 0 | 2325 | 34.54 | 48/48 | - | - | - |
| agents | extent | 6 (0) | 0 | 0 | 0 | 1773 | 33.96 | 48/48 | - | - | - |
| agents | no_read | 6 (0) | 2515323 | 2305906 | 2672576 | 2274 | 33.97 | 48/48 | - | - | - |
| agents | no_populate | 6 (0) | 37295 | 36666 | 38046 | 1890 | 33.33 | 48/48 | - | - | - |
| agents | cow | 6 (0) | 563993 | 562196 | 565524 | 2067 | 33.71 | 48/48 | - | - | - |
| agents | cow_noread | 6 (0) | 7410737 | 7396178 | 7433726 | 4998 | 33.49 | 48/48 | - | - | - |
| publisher | fork_copy | 6 (0) | 0 | 0 | 0 | - | - | - | 0 | 553.43 | 57.2 |
| publisher | fork_extent | 6 (0) | 0 | 0 | 0 | - | - | - | 0 | 10.74 | 57.8 |
| publisher | fork_refill | 6 (0) | 0 | 0 | 0 | - | - | - | 0 | 15.25 | 57.9 |
