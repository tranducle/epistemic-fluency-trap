# Reproducibility

The repository separates evidence verification from full raw-data recomputation.

## Environment

Use Python 3.9 with the package versions in `requirements.txt`. The recorded local environment and model runtime are in `manifests/environment.json`, `manifests/hardware.json`, and `manifests/models.json`.

## Automated tests

```bash
PYTHONPATH=src python -m pytest -q tests
```

## Verify released checksums

```bash
shasum -a 256 -c RELEASE_SHA256.txt
```

## Inspect locked evidence

Actual instruction-model responses and conditioned scenarios are under `outputs/llm/`. Aggregate metrics and statistical summaries are under `results/`. Row-level split manifests and prediction arrays are regenerated from the source datasets and are not committed to the public branch.

The records under `verification/` document provenance, split integrity, baseline sanity, method mechanism, ablations, robustness, and result integrity.

## Reading the verification records

Each `input_artifacts` field records paths from the full experiment workspace at the time the check ran. Raw datasets, row-level CSV files, and prediction arrays are not part of this public tree. A `PASS` value reports the result of the recorded experiment check. It does not mean that every generation-time input is included in the current checkout.

## Full raw-data execution

After acquiring the source datasets described in `DATA.md`, place them under `data/` and run the required stage:

```bash
python experiments/run_pipeline.py splits
python experiments/run_pipeline.py salad
python experiments/run_pipeline.py lmd
python experiments/run_pipeline.py stimuli
python experiments/run_pipeline.py governance
```

The instruction-model stage also requires Ollama and the exact model versions listed in `manifests/models.json`:

```bash
python experiments/run_pipeline.py llm
```

Full recomputation can be resource intensive because the raw SALAD-SOC and LMD files are large. The included aggregate result files and verification records support claim auditing, while full row-level regeneration requires the source datasets.
