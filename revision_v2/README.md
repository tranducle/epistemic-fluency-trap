# JMIS Evidence Pipeline v2

This directory replaces the legacy row-random evaluation with an evidence-gated workflow.

## Scientific boundary

The pipeline tests three claims only:

1. A structural verifier can be evaluated on the official SALAD train, validation, and test partitions without using label-bearing fields.
2. A joint structural and text classifier can be stress tested with fixed appended text attacks while preserving the original event summary.
3. Small open instruction models can be evaluated for model-specific disposition changes under fixed indirect instruction prompts and a structural counterevidence cue.

The pipeline does not claim a human verification effect. It does not treat model outputs as evidence about analyst cognition. Human effects require a separately approved and preregistered study.

## Gate order

1. G0 checks source, schema, row availability, and label diversity.
2. G1 checks official split preservation, identifier overlap, and forbidden feature exclusion.
3. G2 checks finite and noncollapsed baseline predictions.
4. G3 checks PITSA score diversity and text-channel invariance.
5. G5 checks fixed attack budgets, original-text preservation, source transfer, and time transfer.
6. G6 records prediction-level provenance, exact paired tests, and cluster bootstrap intervals.
7. G8 is produced only after the actual LLM pilot is aggregated.

## Main outputs

- `results/salad_official_results.json`
- `results/predictions_v2.parquet`
- `results/llm_sample.jsonl.gz`
- `results/llm_*_summary.json`
- `results/combined_results.json`
- `generated/results_macros.tex`
- `generated/table_salad_v2.tex`
- `generated/table_llm_v2.tex`

All manuscript numbers must be imported from the generated artifacts. Manual transcription is not permitted.
