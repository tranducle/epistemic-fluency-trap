# Data access

Raw datasets are not included. Acquire them from the original providers and verify the local files against `manifests/data_manifest.json`.

## SALAD-SOC

Source repository: `https://github.com/nutthakorn7/SALAD-SOC.git`

Pinned source commit: `07ede61490b99363bf30d2f36a81da28ec6a4a51`

Expected local paths:
- `data/salad_soc/salad_train.csv`
- `data/salad_soc/salad_val.csv`
- `data/salad_soc/salad_test.csv`

The manifest records SHA-256 checksums and byte counts for all three official partitions.

## Lateral Movement Dataset

Source repository: `https://github.com/ChristosSmiliotopoulos/Lateral-Movement-Dataset%2D%2DLMD_Collections.git`

Pinned source commit: `3eb643d7b6c9d2c6ba50933a427fc825cc509336`

Expected local path:
`data/lmd_2023/full-csv(Evaluated Titles).csv`

Label semantics used by the analysis are 0 Normal, 1 EoRS, and 2 EoHT.

## OTRF Security-Datasets

Source repository: `https://github.com/OTRF/Security-Datasets.git`

Pinned source commit: `d9d40ef123d2c87d5d3df28c96bcab4f0faccc87`

OTRF is used to construct attack-conditioned scenarios. It is not treated as independently labeled population ground truth for classifier accuracy.

## CISA Known Exploited Vulnerabilities

CISA KEV is used as deployment context only. The local checksum is recorded in `manifests/data_manifest.json`.
