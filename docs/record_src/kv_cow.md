| Agents | Way to obtain the prefix | Runs (failed agents) | Texts equal to copy | Decode of the agent's own task (ms) | First token after process start (ms) | Generation against copy | Memory of the agents (MiB) | Per agent above extents (MiB) | Prefix pages the agents map: resident sum / proportional sum (MiB) |
|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | copy from the state file (upstream) | 6 (0) | 6/6 | 104 | 999 | - | 2549 | 1812 | - |
| 1 | copy-on-write, 2 MiB file, CPU read pass | 6 (0) | 6/6 | 178 | 854 | 0.961x [0.933, 0.989] | 892 | 154 | 910 / 910 |
| 1 | copy-on-write, 2 MiB file, no read pass | 6 (0) | 6/6 | 654 | 1339 | 0.953x [0.926, 0.979] | 1648 | 911 | 910 / 910 |
| 1 | copy-on-write, 4 KiB file, CPU read pass | 6 (0) | 6/6 | 473 | 1160 | 0.959x [0.927, 0.992] | 828 | 90 | 910 / 910 |
| 1 | copy-on-write, 4 KiB file, no read pass | 6 (0) | 6/6 | 701 | 1393 | 0.957x [0.921, 0.995] | 1632 | 894 | 910 / 910 |
| 1 | extent, tail follows use | 6 (0) | 6/6 | 110 | 790 | 0.967x [0.936, 0.999] | 738 | 0 | 892 / 892 |
| 8 | copy from the state file (upstream) | 6 (0) | 48/48 | 339 | 2181 | - | 20261 | 1815 | - |
| 8 | copy-on-write, 2 MiB file, CPU read pass | 6 (0) | 48/48 | 595 | 2001 | 0.974x [0.953, 0.996] | 6704 | 120 | 7224 / 1742 |
| 8 | copy-on-write, 2 MiB file, no read pass | 6 (0) | 48/48 | 3511 | 4941 | 0.967x [0.956, 0.977] | 12992 | 907 | 7224 / 7224 |
| 8 | copy-on-write, 4 KiB file, CPU read pass | 6 (0) | 48/48 | 2469 | 3923 | 0.957x [0.948, 0.966] | 6720 | 122 | 7224 / 1737 |
| 8 | copy-on-write, 4 KiB file, no read pass | 6 (0) | 48/48 | 3727 | 5161 | 0.967x [0.957, 0.976] | 12990 | 906 | 7224 / 7224 |
| 8 | extent, tail follows use | 6 (0) | 48/48 | 322 | 1735 | 0.980x [0.973, 0.988] | 5740 | 0 | 7140 / 1036 |
