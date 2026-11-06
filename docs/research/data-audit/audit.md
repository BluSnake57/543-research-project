# Evaluation data audit

Status: passed

| Case | Status | Final nodes | First-hour nodes / edges | Repairs |
|---|---|---:|---:|---|
| case-01 | passed | 564 | 135 / 136 | {'self_edge': 3, 'duplicate': 45, 'backward-time': 10} |
| case-02 | passed | 880 | 420 / 421 | {'duplicate': 25, 'backward-time': 4} |
| case-03 | passed | 953 | 364 / 367 | {'self_edge': 12, 'backward-time': 18, 'duplicate': 62} |
| case-04 | passed | 648 | 368 / 371 | {'self_edge': 3, 'duplicate': 44, 'backward-time': 6} |
| case-05 | passed | 188 | 66 / 68 | {'backward-time': 9, 'duplicate': 19} |
| case-06 | passed | 259 | 210 / 209 | {'duplicate': 2} |
| case-07 | passed | 124 | 89 / 88 | {'duplicate': 3, 'backward-time': 1} |
| case-08 | passed | 316 | 65 / 64 | {'duplicate': 1} |
| case-09 | passed | 1596 | 477 / 480 | {'backward-time': 17, 'duplicate': 23} |
| case-10 | passed | 630 | 577 / 580 | {'duplicate': 2} |
| case-11 | passed | 873 | 336 / 336 | {'backward-time': 1, 'duplicate': 4} |
| case-12 | passed | 999 | 661 / 664 | {'duplicate': 17, 'backward-time': 3} |
| case-13 | passed | 128 | 91 / 91 | {'backward-time': 2, 'duplicate': 4} |
| case-14 | passed | 180 | 169 / 168 | {'duplicate': 3, 'backward-time': 1} |
| case-15 | passed | 263 | 116 / 118 | {'self_edge': 3, 'backward-time': 3, 'duplicate': 9} |

## Overlap eligibility

- case-03: excluded from held-out metrics. Exact text matches: []; near matches: ['715253497462145024'].
- case-12: excluded from held-out metrics. Exact text matches: ['693131575261659137']; near matches: [].

Pending human event-level review; lexical matching is not semantic adjudication.

