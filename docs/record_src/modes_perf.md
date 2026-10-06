| Compute shared by | Tenants | Read at the same time, copies (ms) | Read at the same time, shared base (ms) | Slowdown against a tenant alone, shared base | Memory, copies (MiB) | Memory, shared base (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| one MIG instance, time-sliced | 2 | 140.0 | 159.8 | 2.15x | 8314 | 4218 |
| one MIG instance, time-sliced | 4 | 281.4 | 321.9 | 4.32x | 16621 | 4333 |
| one MIG instance, time-sliced | 8 | 572.6 | 646.0 | 8.68x | 33234 | 4562 |
| one MIG instance each (alternating) | 2 | 82.7 | 90.4 | 1.01x | 8314 | 4218 |
| one MIG instance each (alternating) | 4 | 174.4 | 193.1 | 2.16x | 16621 | 4333 |
| one MIG instance each (alternating) | 8 | 351.6 | 389.0 | 4.35x | 33234 | 4563 |
| one MIG instance, one MPS server | 2 | 86.9 | 91.3 | 1.23x | 8296 | 4200 |
| one MIG instance, one MPS server | 4 | 144.5 | 148.8 | 2.00x | 16588 | 4300 |
| one MIG instance, one MPS server | 8 | 270.3 | 278.0 | 3.74x | 33170 | 4498 |
| two MIG instances, an MPS server in each | 2 | 82.7 | 90.4 | 1.01x | 8295 | 4199 |
| two MIG instances, an MPS server in each | 4 | 107.0 | 112.1 | 1.25x | 16587 | 4299 |
| two MIG instances, an MPS server in each | 8 | 178.6 | 187.1 | 2.09x | 33169 | 4497 |
