# Usage Guide — Data Observatory | Moscoso_80

> For: Statistical Learning Lab | Universidad de Antioquia 2026-1
> Files: `runner.py` + `agent_prompt.md`

---

## Prerequisites

Install dependencies once:

```bash
pip install pandas numpy matplotlib seaborn scipy
```

---

## Repository Structure

```
semestre2026-1/
└── Estudiantes/
    └── Moscoso_80/
        ├── runner.py            ← shared; never edit per-lab
        ├── agent_prompt.md      ← shared agent instructions
        ├── usage_guide.md       ← this file
        └── Laboratorios/
            ├── Lab_01_pinguinos/
            │   ├── student_log.md          ← YOU write this
            │   └── artifacts/
            │       ├── run_log.json        ← audit trail (never delete)
            │       ├── 00_raw_profile__OBSERVE__*.json
            │       ├── 01_schema__OBSERVE__*.json
            │       ├── ...
            │       └── plots/
            │
            └── Lab_02_pandas/
                ├── student_log.md          ← shared across both datasets
                └── artifacts/
                    ├── run_log.json        ← shared audit trail for the lab
                    ├── eurusd/             ← dataset-label subfolder
                    │   ├── 00_raw_profile__OBSERVE__*.json
                    │   └── plots/
                    └── breast_cancer/      ← dataset-label subfolder
                        ├── 00_raw_profile__OBSERVE__*.json
                        └── plots/
```

> **Artifacts are never overwritten.** Each run appends `__PHASE__TIMESTAMP`.
> Always use the most recent version when copying to the agent.

---

## Runner Command Reference

### Single-dataset lab (e.g. Lab 01 — Penguins)

```bash
# OBSERVE phase
python runner.py \
  --lab Lab_01_pinguinos \
  --phase OBSERVE \
  --origin seaborn \
  --dataset penguins

# DESCRIBE phase
python runner.py \
  --lab Lab_01_pinguinos \
  --phase DESCRIBE \
  --origin seaborn \
  --dataset penguins

# HYPOTHESIZE phase — run only the specific operations the agent requested
python runner.py \
  --lab Lab_01_pinguinos \
  --phase HYPOTHESIZE \
  --origin seaborn \
  --dataset penguins \
  --ops "crosstab(species,island)" "plot_box(sex,body_mass_g)"
```

> **`--ops` syntax:** each operation is a string `name(arg1,arg2,...)`. Pass
> multiple operations as space-separated quoted strings after `--ops`.
>
> Supported operations in HYPOTHESIZE phase:
> ```
> crosstab(col_a, col_b)
> correlation_matrix(method)          # method: pearson | spearman | kendall
> plot_count(x)                       # or: plot_count(x, hue)
> plot_hist(x)                        # or: plot_hist(x, groupby)
> plot_box(x, y)
> plot_scatter(x, y)                  # or: plot_scatter(x, y, hue)
> plot_heatmap_corr(method)
> ```

### Multi-dataset lab (e.g. Lab 02 — two remote CSV datasets)

```bash
# EUR/USD — OBSERVE
python runner.py \
  --lab Lab_02_pandas \
  --dataset-label eurusd \
  --phase OBSERVE \
  --origin url \
  --dataset https://raw.githubusercontent.com/hernansalinas/Curso_aprendizaje_estadistico/main/datasets/Pandas_data_historical_dataEURUSD.csv

# EUR/USD — DESCRIBE
python runner.py \
  --lab Lab_02_pandas \
  --dataset-label eurusd \
  --phase DESCRIBE \
  --origin url \
  --dataset https://raw.githubusercontent.com/hernansalinas/Curso_aprendizaje_estadistico/main/datasets/Pandas_data_historical_dataEURUSD.csv

# Breast Cancer — OBSERVE
python runner.py \
  --lab Lab_02_pandas \
  --dataset-label breast_cancer \
  --phase OBSERVE \
  --origin url \
  --dataset https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
```

### Local CSV

```bash
python runner.py \
  --lab Lab_03_custom \
  --phase OBSERVE \
  --origin csv \
  --dataset path/to/my_data.csv
```

---

## Step-by-Step Workflow

The workflow is identical for every lab. Replace `{LAB}` and `{DATASET_LABEL}`
with the values you are using.

---

### Step 0 — Write your pre-analysis (do this FIRST, before any code)

Open Claude and paste the full contents of `agent_prompt.md` at the start of the
conversation. Then send:

```
Current lab:      {LAB}
Dataset label:    none
Current phase:    STUDENT_PREANALYSIS

Contents of available artifacts: (none yet)
```

Claude will ask you 4 questions. Write your answers in:

```
Laboratorios/{lab}/student_log.md
```

The runner creates a blank template on the first run, but you can also create it
manually. Use this format:

```markdown
# Student Log

## Phase 0 — Pre-Analysis

**Dataset:** penguins

1. What is the dataset about? Each row is a penguin observation...
2. What are you curious about? Whether species differ in beak shape...
3. Which variables do you think are related? flipper_length and body_mass...
4. What would surprise you? If island and species were independent...
```

---

### Step 1 — Run OBSERVE

```bash
python runner.py --lab {LAB} [--dataset-label {LABEL}] --phase OBSERVE --origin {ORIGIN} --dataset {DATASET}
```

This creates 4 artifact files. Then go to Claude and paste their contents:

```
Current lab:   Lab_01_pinguinos
Dataset label: none
Current phase: OBSERVE

Contents of available artifacts:
--- 00_raw_profile__OBSERVE__*.json ---
{ ...paste content... }

--- 01_schema__OBSERVE__*.json ---
{ ...paste content... }

--- 02_missing_report__OBSERVE__*.json ---
{ ...paste content... }

--- 03_duplicates_report__OBSERVE__*.json ---
{ ...paste content... }
```

Read every `rationale` in the agent's response — that is the learning content.
Answer any `socratic_questions` before moving on.

---

### Step 2 — Run DESCRIBE

```bash
python runner.py --lab {LAB} [--dataset-label {LABEL}] --phase DESCRIBE --origin {ORIGIN} --dataset {DATASET}
```

This generates statistics and up to 10 plots. Then paste into Claude:
- `04a_numeric_summary__DESCRIBE__*.json`
- `04b_categorical_summary__DESCRIBE__*.json`
- `04d_correlation_pearson__DESCRIBE__*.json`
- `05_visual_registry__DESCRIBE__*.json`

Claude will ask `socratic_questions` about the plots. Answer them before asking
it to generate hypotheses — this is intentional.

---

### Step 3 — Hypothesize and Conclude

**3a — Send available artifacts to the agent:**

```
Current lab:   Lab_01_pinguinos
Dataset label: none
Current phase: HYPOTHESIZE_AND_CONCLUDE

Contents of available artifacts:
...paste 04 and 05 artifacts...

Also available:
--- student_log.md ---
...paste your full student_log.md here...
```

The agent will ask 2–3 socratic questions first — answer them before it
generates any hypotheses. This is intentional.

**3b — Run additional operations (if the agent requests them):**

The agent may request specific extra operations (e.g. a `crosstab` or a new
plot) that were not produced in DESCRIBE. Run them with `--phase HYPOTHESIZE`
and `--ops` instead of re-running the full DESCRIBE pipeline:

```bash
python runner.py \
  --lab {LAB} [--dataset-label {LABEL}] \
  --phase HYPOTHESIZE \
  --origin {ORIGIN} \
  --dataset {DATASET} \
  --ops "crosstab(species,island)" \
        "crosstab(species,sex)" \
        "plot_scatter(bill_depth_mm,bill_length_mm,species)" \
        "plot_box(sex,body_mass_g)"
```

New artifacts are written to the same `artifacts/` folder with the
`__HYPOTHESIZE__TIMESTAMP` suffix. Paste their contents back to the agent.

**3c — Collect final output:**

The agent produces `06_hypotheses_log.json`, `07_conclusions.json`, and
`report.md`. Copy those JSON blocks from the response into files in your
`artifacts/` folder.

---

## Multi-Dataset Labs — Key Differences

When a lab has multiple datasets (like Lab 02):

1. Run the full workflow separately for each dataset using `--dataset-label`.
2. Use a **separate conversation** with the agent for each dataset.
3. Paste the **same** `student_log.md` in both conversations (Phase 0 covers the lab, not a single dataset).
4. `run_log.json` is shared — check it anytime to see everything that was run.

---

## Approve / Reject Workflow

| Decision | What to do |
|---|---|
| ✅ **Approve** | Run the operations in `actions_to_run` exactly as specified. |
| ✏️ **Modify** | Tell the agent what to change and why, request a revised JSON. |
| ❌ **Reject** | Explain what is wrong and ask the agent to rethink. |

You are the scientific decision-maker. The agent is your planning assistant.

---

## Naming Conventions

**Lab folders** — consistent pattern:
```
Lab_01_pinguinos
Lab_02_pandas
Lab_03_regresion
```

**Dataset labels** — short, lowercase, no spaces:
```
eurusd  |  breast_cancer  |  titanic  |  iris
```

**Artifact files** — auto-generated:
```
{stem}__{PHASE}__{YYYYMMDD_HHMMSS}.json
heatmap_corr_pearson__DESCRIBE__20260315_143601.png
```

---

## Common Errors

**`ModuleNotFoundError`**
→ `pip install seaborn matplotlib pandas numpy scipy`

**`FileNotFoundError` on CSV**
→ The path is relative to where you run the command, not the script location.

**URL dataset fails to load**
→ Check internet connection. Verify the URL points directly to a raw CSV file.

**`JSONDecodeError` in run_log**
→ Delete `artifacts/run_log.json` and re-run. Happens only if a run was interrupted mid-write.

**Agent invents numbers**
→ You forgot to paste artifact contents into `{ARTIFACTS_CONTENT}`.

**Wrong artifact folder**
→ Check that `--lab` and `--dataset-label` match what you used in previous runs.
Check `run_log.json` if unsure.

---

## Reproducibility Checklist

- [ ] `random_seed = 42` is unchanged in `runner.py`.
- [ ] All artifacts are inside `Laboratorios/{lab}/`.
- [ ] `run_log.json` exists and was not manually edited.
- [ ] `student_log.md` has a Phase 0 entry.
- [ ] Plots are in the correct `plots/` subfolder with descriptive names.
- [ ] No artifact file was manually overwritten.
- [ ] Multi-dataset labs use consistent `--dataset-label` across all phases.