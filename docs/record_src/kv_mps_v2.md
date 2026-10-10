| Sharing | Mode | Runs (failed processes) | Pause of the parent (ms) | State file (MiB) | Attach (ms) | First token (ms) | Children (tokens/s) | Children memory (MiB) | Prefix PSS (MiB) | Texts equal to copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one MIG instance, time-sliced | copy | 6 (0) | 551.30 | 892.80 | 535.9 | 2533 | 21.45 | 20287 | 0 | 48/48 |
| one MIG instance, time-sliced | extent | 6 (0) | 11.28 | 0.25 | 53.8 | 2053 | 21.45 | 5895 | 913 | 48/48 |
| one MIG instance, one MPS server | copy | 6 (0) | 545.74 | 892.80 | 531.1 | 2379 | 24.13 | 20197 | 0 | 48/48 |
| one MIG instance, one MPS server | extent | 6 (0) | 10.93 | 0.25 | 53.3 | 1871 | 24.02 | 5832 | 916 | 48/48 |
| one MIG instance each (alternating) | copy | 6 (0) | 555.78 | 892.80 | 487.7 | 2275 | 32.40 | 20228 | 0 | 48/48 |
| one MIG instance each (alternating) | extent | 6 (0) | 11.08 | 0.25 | 56.4 | 1809 | 31.58 | 5843 | 1020 | 48/48 |
| two MIG instances, an MPS server in each | copy | 6 (0) | 556.01 | 892.80 | 493.0 | 2121 | 36.37 | 20192 | 0 | 48/48 |
| two MIG instances, an MPS server in each | extent | 6 (0) | 11.24 | 0.25 | 51.0 | 1656 | 35.13 | 5796 | 939 | 48/48 |
