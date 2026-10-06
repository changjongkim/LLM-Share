| Compute shared by | Agents | Generation, device copy (tokens/s, sum) | Generation, in place (tokens/s, sum) | In place / device copy | Memory, device copy (MiB) | Memory, in place (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| one MIG instance, time-sliced | 1 | 28.5 +/- 0.1 | 27.7 +/- 0.1 | 0.971x | 5259 | 5198 |
| one MIG instance, time-sliced | 2 | 26.5 +/- 0.0 | 25.6 +/- 0.0 | 0.966x | 10462 | 5941 |
| one MIG instance, time-sliced | 4 | 26.6 +/- 0.0 | 25.4 +/- 0.0 | 0.956x | 20975 | 7366 |
| one MIG instance, time-sliced | 8 | 26.7 +/- 0.1 | 25.3 +/- 0.0 | 0.946x | 41921 | 10316 |
| one MIG instance each (alternating) | 1 | 28.5 +/- 0.0 | 27.6 +/- 0.0 | 0.969x | 5259 | 5178 |
| one MIG instance each (alternating) | 2 | 44.8 +/- 0.1 | 44.1 +/- 0.0 | 0.986x | 10430 | 5937 |
| one MIG instance each (alternating) | 4 | 41.4 +/- 0.2 | 41.0 +/- 0.0 | 0.990x | 20925 | 7283 |
| one MIG instance each (alternating) | 8 | 41.4 +/- 0.2 | 40.4 +/- 0.1 | 0.974x | 41798 | 10236 |
| one MIG instance, one MPS server | 1 | 28.6 +/- 0.1 | 27.7 +/- 0.1 | 0.967x | 5283 | 5148 |
| one MIG instance, one MPS server | 2 | 29.3 +/- 0.0 | 28.3 +/- 0.0 | 0.966x | 10456 | 5851 |
| one MIG instance, one MPS server | 4 | 29.6 +/- 0.0 | 28.6 +/- 0.0 | 0.967x | 20921 | 7313 |
| one MIG instance, one MPS server | 8 | 29.4 +/- 0.1 | 28.6 +/- 0.0 | 0.971x | 41822 | 10228 |
| two MIG instances, an MPS server in each | 1 | 28.5 +/- 0.1 | 27.6 +/- 0.1 | 0.970x | 5236 | 5174 |
| two MIG instances, an MPS server in each | 2 | 44.8 +/- 0.1 | 44.1 +/- 0.1 | 0.985x | 10452 | 5794 |
| two MIG instances, an MPS server in each | 4 | 45.7 +/- 0.1 | 45.1 +/- 0.0 | 0.986x | 20847 | 7218 |
| two MIG instances, an MPS server in each | 8 | 45.9 +/- 0.2 | 45.4 +/- 0.0 | 0.989x | 41680 | 10138 |
