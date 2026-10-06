| Configuration | Runs (failed servers) | Generation, all agents (tokens/s) | 12-SM server | 6-SM server | First server ready (s) | All agents ready (s) | Hand-over (ms) | State file (MiB) | Memory of all servers (MiB) | Texts equal to the copy configuration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one server in the 12-SM instance, 8 sequences | 6 (0) | 75.6 | 75.6 | - | 20.7 | 20.7 | - | - | 2578 | 36/48 |
| a server in each instance, 4 sequences each; each computes the prefix | 6 (0) | 106.2 | 61.7 | 44.5 | 20.9 | 30.6 | - | - | 5081 | 36/48 |
| a server in each instance; the second copies the state file | 6 (0) | 105.0 | 61.4 | 43.7 | 21.2 | 22.2 | 565.4 | 892.80 | 5101 | 48/48 |
| a server in each instance; the second maps the published prefix | 6 (0) | 102.9 | 60.9 | 41.9 | 20.7 | 21.4 | 8.8 | 0.25 | 2437 | 48/48 |
