# Replication Run — Blood Cell Anomaly Dataset

This folder contains a second execution of the same agent workflow applied to a different dataset (blood cell anomaly classification, 5 880 rows), demonstrating that the pipeline is replicable across datasets without modification.

## Contents

- `runner.ipynb` — same notebook runner, different dataset loaded
- `artifacts/` — full evidence set: profile, descriptive stats, visual registry, hypotheses, conclusions, statistical tests, report
  - `08_tests.json` — includes Mann-Whitney U, Welch t-test, Levene, and chi-squared results

## Key difference vs main run

The blood cell dataset includes a binary anomaly label, so Phase 3 also produced `08_tests.json` with group-difference tests, exercising the optional tests section of the agent spec.
