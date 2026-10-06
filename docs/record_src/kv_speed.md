| MIG instance | Prefix (tokens) | Tokens generated | Processes | Cache | Runs (failed) | Generation, sum (tokens/s) | Against device memory |
|---|---:|---:|---:|---|---:|---:|---|
| 12-SM | 4081 | 256 | 1 | device memory, prefix computed (upstream) | 6 (0) | 25.28 | - |
| 12-SM | 4081 | 256 | 1 | private host memory, 2 MiB pages, prefix computed | 6 (0) | 25.29 | 1.000x [0.998, 1.003] |
| 12-SM | 4081 | 256 | 1 | private host memory that follows use, 4 KiB pages, prefix computed | 6 (0) | 25.01 | 0.989x [0.987, 0.991] |
| 12-SM | 4081 | 256 | 1 | device memory, prefix copied from the state file | 6 (0) | 25.27 | 1.000x [0.997, 1.003] |
| 12-SM | 4081 | 256 | 1 | mapped prefix, private tail on 2 MiB pages | 6 (0) | 25.18 | 0.996x [0.993, 0.999] |
| 12-SM | 4081 | 256 | 1 | mapped prefix, private tail that follows use | 6 (0) | 25.17 | 0.996x [0.992, 0.999] |
| 12-SM | 4081 | 256 | 2 | device memory, prefix computed (upstream) | 6 (0) | 24.31 | - |
| 12-SM | 4081 | 256 | 2 | private host memory, 2 MiB pages, prefix computed | 6 (0) | 24.43 | 1.005x [1.004, 1.006] |
| 12-SM | 4081 | 256 | 2 | private host memory that follows use, 4 KiB pages, prefix computed | 6 (0) | 24.22 | 0.996x [0.996, 0.997] |
| 12-SM | 4081 | 256 | 2 | device memory, prefix copied from the state file | 6 (0) | 24.32 | 1.000x [0.999, 1.002] |
| 12-SM | 4081 | 256 | 2 | mapped prefix, private tail on 2 MiB pages | 6 (0) | 24.38 | 1.003x [1.002, 1.003] |
| 12-SM | 4081 | 256 | 2 | mapped prefix, private tail that follows use | 6 (0) | 24.35 | 1.002x [1.001, 1.003] |
| 12-SM | 16321 | 128 | 1 | device memory, prefix computed (upstream) | 6 (0) | 23.89 | - |
| 12-SM | 16321 | 128 | 1 | private host memory, 2 MiB pages, prefix computed | 6 (0) | 23.98 | 1.004x [0.998, 1.009] |
| 12-SM | 16321 | 128 | 1 | private host memory that follows use, 4 KiB pages, prefix computed | 6 (0) | 23.91 | 1.001x [1.000, 1.002] |
| 12-SM | 16321 | 128 | 1 | device memory, prefix copied from the state file | 6 (0) | 23.94 | 1.002x [0.998, 1.006] |
| 12-SM | 16321 | 128 | 1 | mapped prefix, private tail on 2 MiB pages | 6 (0) | 23.94 | 1.002x [0.998, 1.006] |
| 12-SM | 16321 | 128 | 1 | mapped prefix, private tail that follows use | 6 (0) | 23.93 | 1.002x [0.997, 1.006] |
| 6-SM | 4081 | 256 | 1 | device memory, prefix computed (upstream) | 6 (0) | 16.26 | - |
| 6-SM | 4081 | 256 | 1 | private host memory, 2 MiB pages, prefix computed | 6 (0) | 15.90 | 0.978x [0.972, 0.984] |
| 6-SM | 4081 | 256 | 1 | private host memory that follows use, 4 KiB pages, prefix computed | 6 (0) | 15.78 | 0.970x [0.968, 0.972] |
| 6-SM | 4081 | 256 | 1 | device memory, prefix copied from the state file | 6 (0) | 16.28 | 1.001x [1.000, 1.002] |
| 6-SM | 4081 | 256 | 1 | mapped prefix, private tail on 2 MiB pages | 6 (0) | 15.90 | 0.978x [0.975, 0.981] |
| 6-SM | 4081 | 256 | 1 | mapped prefix, private tail that follows use | 6 (0) | 15.88 | 0.976x [0.974, 0.979] |
| 6-SM | 16321 | 128 | 1 | device memory, prefix computed (upstream) | 6 (0) | 15.85 | - |
| 6-SM | 16321 | 128 | 1 | private host memory, 2 MiB pages, prefix computed | 6 (0) | 14.95 | 0.943x [0.939, 0.948] |
| 6-SM | 16321 | 128 | 1 | private host memory that follows use, 4 KiB pages, prefix computed | 6 (0) | 14.91 | 0.941x [0.938, 0.943] |
| 6-SM | 16321 | 128 | 1 | device memory, prefix copied from the state file | 6 (0) | 15.89 | 1.002x [1.000, 1.005] |
| 6-SM | 16321 | 128 | 1 | mapped prefix, private tail on 2 MiB pages | 6 (0) | 14.93 | 0.942x [0.940, 0.944] |
| 6-SM | 16321 | 128 | 1 | mapped prefix, private tail that follows use | 6 (0) | 14.93 | 0.942x [0.940, 0.945] |
