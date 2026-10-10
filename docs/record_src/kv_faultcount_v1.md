| Processes | Mode | Runs (failed) | Pages faulted in for the GPU | lowest | highest | First token (ms) | Throughput (tokens/s) | Texts equal to copy | Pages before the publish | Publish (ms) | Next task (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agents | copy | 6 (0) | 0 | 0 | 0 | 2350 | 34.48 | 48/48 | - | - | - |
| agents | extent | 6 (0) | 0 | 0 | 0 | 1761 | 34.02 | 48/48 | - | - | - |
| agents | no_read | 6 (0) | 2522352 | 2368218 | 2651914 | 2298 | 33.94 | 48/48 | - | - | - |
| agents | no_populate | 6 (0) | 37123 | 36570 | 37682 | 1925 | 33.28 | 48/48 | - | - | - |
| agents | cow | 6 (0) | 564685 | 563744 | 565210 | 2078 | 33.65 | 48/48 | - | - | - |
| agents | cow_noread | 6 (0) | 7420080 | 7398896 | 7443028 | 4886 | 33.59 | 48/48 | - | - | - |
| publisher | fork_copy | 6 (0) | 0 | 0 | 0 | - | - | - | 0 | 549.08 | 57.3 |
| publisher | fork_extent | 6 (0) | 113357 | 113348 | 113364 | - | - | - | 0 | 9.19 | 135.7 |
| publisher | fork_refill | 6 (0) | 0 | 0 | 0 | - | - | - | 0 | 13.91 | 57.8 |
