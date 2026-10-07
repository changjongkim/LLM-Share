| Stack | Groups x workers x turns | Runs (failed) | Prefix (tokens) | Completion (s) | 95% CI | vs stock | Memory (MiB) | Files (MiB) | Worker attach (ms) | Worker first token (ms) | Energy, input rail (J) | prefix / leaders / workers (J) | vs stock | GPU rail (J) | Worker texts equal to stock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | 2x4x3 | 6 (0) | 11788 | 80.3 | 0.2 | 1.000 | 67384 | 0 | 376.3 | 6305 | 5906 | 833 / 394 / 4679 | 1.000 | 2545 | 48/48 |
| copy | 2x4x3 | 6 (0) | 11788 | 77.9 | 0.1 | 0.970 | 17635 | 0 | 380.0 | 2312 | 5834 | 832 / 373 / 4630 | 0.988 | 2580 | 48/48 |
| chain | 2x4x3 | 6 (0) | 11788 | 77.9 | 0.1 | 0.971 | 8577 | 896 | 36.3 | 1926 | 5880 | 859 / 359 / 4662 | 0.996 | 2602 | 48/48 |
