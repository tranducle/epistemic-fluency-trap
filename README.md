# Prompt-isolated structural contestability experiments

This repository contains code and locked evidence artifacts for evaluating a structural verification channel in AI-assisted security triage. The verifier uses structural telemetry without consuming attacker-reachable text. The experiments keep three questions separate: clean predictive utility, invariance to text-only perturbations, and behavior under source or environment shift.

The primary classifier evaluation uses the official SALAD-SOC train, validation, and test partitions. LMD is used for signature, time, and station shift diagnostics. OTRF security datasets are used only to construct attack-conditioned text-channel scenarios. CISA KEV is context data rather than detection ground truth.

The repository includes aggregate result files, actual instruction-model responses for the evaluated scenarios, and machine-readable scientific verification records. Raw source datasets are not redistributed.

## Quick check

Create an environment with the versions in `requirements.txt`, then run:

```bash
PYTHONPATH=src python -m pytest -q tests
```

The test suite checks score computation, feature handling, split logic, and structured model-response parsing.

## Repository layout

- `src/`: analysis and experiment modules.
- `experiments/`: command-line experiment runner.
- `configs/`: experiment configuration.
- `manifests/`: dataset, split, model, hardware, and environment provenance.
- `outputs/`: prediction-level and instruction-model evidence needed for verification.
- `results/`: locked quantitative result files.
- `verification/`: scientific gate records.
- `tests/`: automated tests.
- `DATA.md`: dataset acquisition and checksum instructions.
- `REPRODUCIBILITY.md`: commands for verifying the released evidence.

## Scope

Text-channel invariance is an architectural property under fixed structural inputs. It does not establish robustness to structural manipulation, sensor compromise, or joint-channel attacks. The instruction-model study is limited to the model versions and conditioned scenarios recorded in the manifests. Human response to contestation cues was not studied.
