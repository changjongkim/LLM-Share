| GPU sharing | Fault | Runs | Agents complete | Others complete | Others with the text of the run without a fault | Writes refused | Error of the write | File intact (runs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| timeslice | none | 6 | 48/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| timeslice | write | 6 | 48/48 | 42/42 | 42/42 | 6/6 | cudaErrorIllegalAddress | 6/6 |
| timeslice | kill | 6 | 42/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| mps | none | 6 | 48/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| mps | write | 6 | 0/48 | 0/42 | 0/42 | 6/6 | cudaErrorIllegalAddress | 6/6 |
| mps | kill | 6 | 0/48 | 0/42 | 0/42 | 0/0 | none | 6/6 |
| mig | none | 6 | 48/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| mig | write | 6 | 48/48 | 42/42 | 42/42 | 6/6 | cudaErrorIllegalAddress | 6/6 |
| mig | kill | 6 | 42/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| mig_mps | none | 6 | 48/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| mig_mps | write | 6 | 24/48 | 24/42 | 24/42 | 6/6 | cudaErrorIllegalAddress | 6/6 |
| mig_mps | kill | 6 | 30/48 | 30/42 | 30/42 | 0/0 | none | 6/6 |
