| Placement | Shared and private device memory composed | Refusal | Read-only access to the shared part | Producer's data unchanged |
|---|---:|---|---|---:|
| two processes in one MIG instance | 6/6 | - | enforced | 6/6 |
| one process in each MIG instance | 0/6 | `UNSUPPORTED_at_import_CUDA_ERROR_NOT_INITIALIZED` | - | 0/0 |
| two clients of one MPS server | 6/6 | - | enforced | 6/6 |
