| Configuration | Victim is | Victims | Finished the read in progress | Read again afterwards | Base intact |
|---|---|---:|---:|---:|---:|
| no MPS | other MIG instance | 6 | 6/6 | 6/6 | 6/6 |
| no MPS | same MIG instance, time-sliced | 12 | 12/12 | 12/12 | 12/12 |
| MPS server in the faulting tenant's instance | other MIG instance | 6 | 6/6 | 6/6 | 6/6 |
| MPS server in the faulting tenant's instance | client of the same MPS server | 12 | 0/12 | 0/12 | 12/12 |
| MPS server in each instance | other MIG instance, client of its own MPS server | 12 | 12/12 | 12/12 | 12/12 |
| MPS server in each instance | client of the same MPS server | 6 | 0/6 | 0/6 | 6/6 |
