| Placement | CUDA IPC | Host page table |
|---|---|---|
| two processes in one MIG instance | works 5/5 | works 5/5 |
| one process in each MIG instance | works 0/5 (`UNSUPPORTED_at_open_cudaErrorInvalidValue`) | works 5/5 |
| two clients of one MPS server | works 5/5 | works 5/5 |
