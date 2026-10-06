| Agents | Way to obtain the prefix | Texts equal to copy | Texts equal to recompute: publisher's instance, other instance | Attach (ms) | First token after process start (ms) | Generation, sum (tokens/s) | Generation against copy | Memory of the agents (MiB) | Prefix pages the agents map: resident sum / proportional sum (MiB) |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | recompute | - | 6/6, - | 28 | 6151 | 20.7 | 0.996x [0.993, 0.999] | 1145 | - |
| 1 | copy from the state file (upstream) | 6/6 | 6/6, - | 81 | 819 | 20.8 | - | 1141 | - |
| 1 | copy-on-write mapping, 2 MiB file | 6/6 | 6/6, - | 24 | 864 | 20.5 | 0.984x [0.973, 0.994] | 931 | 238 / 238 |
| 1 | copy-on-write mapping, 4 KiB file | 6/6 | 6/6, - | 27 | 1158 | 20.4 | 0.979x [0.965, 0.994] | 826 | 238 / 238 |
| 1 | extent, whole tail allocated | 6/6 | 6/6, - | 32 | 775 | 20.4 | 0.980x [0.971, 0.989] | 913 | 223 / 223 |
| 1 | extent, tail follows use | 6/6 | 6/6, - | 25 | 761 | 20.4 | 0.981x [0.973, 0.989] | 698 | 223 / 223 |
| 1 | extent, tail follows use, 4 KiB file | 6/6 | 6/6, - | 27 | 776 | 20.4 | 0.978x [0.967, 0.989] | 726 | 223 / 223 |
| 4 | recompute | - | 12/12, 12/12 | 39 | 13199 | 39.1 | 1.014x [1.001, 1.027] | 4556 | - |
| 4 | copy from the state file (upstream) | 24/24 | 12/12, 0/12 | 122 | 1274 | 38.6 | - | 4586 | - |
| 4 | copy-on-write mapping, 2 MiB file | 24/24 | 12/12, 0/12 | 36 | 1493 | 37.7 | 0.977x [0.964, 0.990] | 3235 | 952 / 616 |
| 4 | copy-on-write mapping, 4 KiB file | 24/24 | 12/12, 0/12 | 34 | 2442 | 37.3 | 0.968x [0.955, 0.980] | 3247 | 952 / 616 |
| 4 | extent, whole tail allocated | 24/24 | 12/12, 0/12 | 40 | 1209 | 37.8 | 0.979x [0.969, 0.989] | 3647 | 892 / 229 |
| 4 | extent, tail follows use | 24/24 | 12/12, 0/12 | 33 | 1217 | 37.6 | 0.974x [0.962, 0.987] | 2814 | 892 / 223 |
| 4 | extent, tail follows use, 4 KiB file | 24/24 | 12/12, 0/12 | 37 | 1189 | 37.6 | 0.974x [0.960, 0.988] | 2808 | 892 / 223 |
| 8 | recompute | - | 24/24, 24/24 | 44 | 25608 | 39.0 | 1.014x [1.007, 1.021] | 9158 | - |
| 8 | copy from the state file (upstream) | 48/48 | 24/24, 12/24 | 161 | 1797 | 38.4 | - | 9169 | - |
| 8 | copy-on-write mapping, 2 MiB file | 48/48 | 24/24, 12/24 | 34 | 2186 | 37.1 | 0.966x [0.960, 0.973] | 6545 | 1904 / 1120 |
| 8 | copy-on-write mapping, 4 KiB file | 48/48 | 24/24, 12/24 | 36 | 4130 | 36.9 | 0.959x [0.951, 0.966] | 6529 | 1904 / 1120 |
| 8 | extent, whole tail allocated | 48/48 | 24/24, 12/24 | 45 | 1706 | 37.3 | 0.971x [0.963, 0.979] | 7320 | 1785 / 228 |
| 8 | extent, tail follows use | 48/48 | 24/24, 12/24 | 41 | 1721 | 37.3 | 0.970x [0.956, 0.984] | 5640 | 1785 / 239 |
| 8 | extent, tail follows use, 4 KiB file | 48/48 | 24/24, 12/24 | 49 | 1788 | 37.0 | 0.961x [0.954, 0.969] | 5629 | 1785 / 223 |
