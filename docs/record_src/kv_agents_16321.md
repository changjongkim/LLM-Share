| Agents | Way to obtain the prefix | Texts equal to copy | Texts equal to recompute: publisher's instance, other instance | Attach (ms) | First token after process start (ms) | Generation, sum (tokens/s) | Generation against copy | Memory of the agents (MiB) | Prefix pages the agents map: resident sum / proportional sum (MiB) |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | recompute | - | 6/6, - | 59 | 25254 | 19.9 | 0.995x [0.990, 1.000] | 2556 | - |
| 1 | copy from the state file (upstream) | 6/6 | 6/6, - | 225 | 964 | 20.0 | - | 2532 | - |
| 1 | copy-on-write mapping, 2 MiB file | 6/6 | 6/6, - | 43 | 856 | 19.1 | 0.954x [0.928, 0.981] | 923 | 910 / 910 |
| 1 | copy-on-write mapping, 4 KiB file | 6/6 | 6/6, - | 52 | 1151 | 19.2 | 0.955x [0.925, 0.986] | 815 | 910 / 910 |
| 1 | extent, whole tail allocated | 6/6 | 6/6, - | 70 | 845 | 19.4 | 0.967x [0.932, 1.004] | 1617 | 892 / 892 |
| 1 | extent, tail follows use | 6/6 | 6/6, - | 43 | 791 | 19.2 | 0.955x [0.923, 0.987] | 743 | 892 / 892 |
| 1 | extent, tail follows use, 4 KiB file | 6/6 | 6/6, - | 52 | 814 | 19.3 | 0.961x [0.927, 0.996] | 744 | 892 / 892 |
| 4 | recompute | - | 12/12, 12/12 | 122 | 55866 | 37.2 | 1.062x [1.059, 1.065] | 10275 | - |
| 4 | copy from the state file (upstream) | 24/24 | 12/12, 9/12 | 396 | 1583 | 35.1 | - | 10147 | - |
| 4 | copy-on-write mapping, 2 MiB file | 24/24 | 12/12, 9/12 | 55 | 1413 | 34.6 | 0.986x [0.984, 0.988] | 3325 | 3612 / 1260 |
| 4 | copy-on-write mapping, 4 KiB file | 24/24 | 12/12, 9/12 | 67 | 2377 | 34.3 | 0.979x [0.975, 0.982] | 3326 | 3612 / 1260 |
| 4 | extent, whole tail allocated | 24/24 | 12/12, 9/12 | 93 | 1288 | 34.8 | 0.992x [0.988, 0.996] | 6441 | 3570 / 892 |
| 4 | extent, tail follows use | 24/24 | 12/12, 9/12 | 57 | 1223 | 34.8 | 0.992x [0.987, 0.997] | 2890 | 3570 / 892 |
| 4 | extent, tail follows use, 4 KiB file | 24/24 | 12/12, 9/12 | 62 | 1242 | 34.7 | 0.988x [0.983, 0.993] | 2892 | 3570 / 917 |
| 8 | recompute | - | 24/24, 24/24 | 152 | 110880 | 37.2 | 1.063x [1.054, 1.071] | 20283 | - |
| 8 | copy from the state file (upstream) | 48/48 | 24/24, 15/24 | 484 | 2205 | 35.0 | - | 20331 | - |
| 8 | copy-on-write mapping, 2 MiB file | 48/48 | 24/24, 15/24 | 58 | 1980 | 34.1 | 0.975x [0.966, 0.984] | 6705 | 7224 / 1736 |
| 8 | copy-on-write mapping, 4 KiB file | 48/48 | 24/24, 15/24 | 83 | 3971 | 33.8 | 0.965x [0.958, 0.972] | 6718 | 7224 / 1738 |
| 8 | extent, whole tail allocated | 48/48 | 24/24, 15/24 | 92 | 1774 | 34.4 | 0.984x [0.972, 0.995] | 12969 | 7140 / 1025 |
| 8 | extent, tail follows use | 48/48 | 24/24, 15/24 | 65 | 1778 | 34.4 | 0.982x [0.974, 0.990] | 5849 | 7140 / 908 |
| 8 | extent, tail follows use, 4 KiB file | 48/48 | 24/24, 15/24 | 75 | 1806 | 34.2 | 0.978x [0.969, 0.986] | 5849 | 7140 / 905 |
