| Route | Attempts | Read / access change | Write refused / succeeded | Write result | Exporter's state intact |
|---|---:|---:|---:|---|---:|
| host, read-only mapping | 12 | 12 reads complete | 12 refused | `cudaErrorIllegalAddress` | 12/12 |
| device VMM import | 6 | 6 access raises succeeded | 6 succeeded | allowed | 0/6 |
