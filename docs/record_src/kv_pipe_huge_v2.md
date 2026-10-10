| Stack | Groups x workers x turns | Runs (failed) | Prefix (tokens) | Completion (s) | 95% CI | vs stock | Memory (MiB) | Files (MiB) | Worker attach (ms) | Worker first token (ms) | Energy, input rail (J) | prefix / leaders / workers (J) | vs stock | GPU rail (J) | Worker texts equal to stock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | 2x4x3 | 6 (0) | 11788 | 80.2 | 0.1 | 1.000 | 67403 | 0 | 375.6 | 6266 | 5943 | 845 / 398 / 4700 | 1.000 | 2557 | 48/48 |
| copy | 2x4x3 | 6 (0) | 11788 | 77.9 | 0.1 | 0.971 | 17633 | 0 | 386.5 | 2314 | 5860 | 834 / 370 / 4656 | 0.986 | 2593 | 48/48 |
| chain | 2x4x3 | 6 (0) | 11788 | 77.9 | 0.0 | 0.972 | 8618 | 896 | 37.7 | 1911 | 5901 | 856 / 358 / 4686 | 0.993 | 2614 | 48/48 |
