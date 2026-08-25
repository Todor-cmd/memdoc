# memdoc

Dataset generation, agent inference, analysis scripts, and raw results for the TU Delft DSAIT MSc thesis **The Effect of Evidence Distribution Across Conversational Memory and Documents on QA Agent Answer Correctness**.

Run every command from the repository root.

## Contents

- [1. Repository structure](#1-repository-structure)
- [2. Environment setup](#2-environment-setup)
- [3. Hugging Face dataset](#3-hugging-face-dataset)
- [4. Dataset generation](#4-dataset-generation)
- [5. Agent inference](#5-agent-inference)
- [6. Analysis](#6-analysis)

## 1. Repository structure


| Path                     | Role                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| `prepare_data/`          | Question split, persona assignment, document export, experiment sample, D-optimal design  |
| `sampling_frame_agents/` | Golden-context and no-context screening agents used to build the reasonable-question pool |
| `session_generation/`    | OpenAI batch jobs that write evidence conversations, plus session audit                   |
| `memory_curation/`       | Per-persona memory corpora assembled from generated sessions                              |
| `agents/`                | QA agent implementations (separate store, MemPalace, corrective RAG, unified store)       |
| `experiment/`            | Harness that runs registered agents over the design matrix                                |
| `analysis/`              | Labeling scripts, model-feature tables, and Quarto/R analyses                             |
| `data/`                  | Intermediate and final artifacts (questions, CSVs, batch jobs, memory, experiment runs)   |
| `environment.yaml`       | Conda environment for Python and notebooks                                                |


`data/` is the working store for generated files. Screening outputs live under `data/golden_context_agent_inferences/` and `data/no_context_agent_inferences/`; experiment traces are `data/experiment_runs/<agent_id>.jsonl`.

## 2. Environment setup

Python and notebooks:

```bash
conda env create -f environment.yaml
conda activate env-memdoc
```

For analysis and agent runs, put this key in a `.env` file at the repo root:

```bash
GROQ_API_KEY=...
```

To re-run dataset construction (evidence-session batches), you also need:

```bash
OPENAI_API_KEY=...
```

R and Quarto setup is more brittle. These commands are a starting point and may need small adjustments:

```bash
conda install -c conda-forge \
  r-base r-lme4 r-emmeans r-car r-readr r-dplyr r-tidyr \
  r-ggplot2 r-broom.mixed r-dharma r-knitr r-jsonlite

Rscript -e 'install.packages("skpr", repos="https://cloud.r-project.org")'
```

Install [Quarto](https://quarto.org/docs/get-started/).

## 3. Hugging Face dataset

The eval set lives at [todor-cmd/memdoc](https://huggingface.co/datasets/todor-cmd/memdoc): 198 multi-hop questions whose gold evidence can sit in conversational memory, in a document corpus, or in both, with the question and evidence held fixed so store-placement effects are not confounded with item difficulty.

**If you only want the dataset** for your own agents or analysis, stop here. Load it from the Hub; you do not need this repository or a pickle:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="todor-cmd/memdoc",
    repo_type="dataset",
    local_dir="memdoc_data",
)
```

The table is `experiment_questions.csv`. Evidence columns (`memory_evidence`, `evidence_list`, `golden_memory_evidence`, `golden_document_evidence`) are JSON lists. Memory JSON and the document corpus are extra files on the same repo.

**If you want to run the agents in this repo**, skip [§4 Dataset generation](#4-dataset-generation), convert that CSV to the pickle the harness expects, and download the stores into `data/`:

The Hub table omits the 12 `null_query` blocks from the thesis 210-question set. The pickle filename below is unchanged so existing defaults still work.

```python
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

HF_DATASET = "todor-cmd/memdoc"
LIST_COLUMNS = [
    "golden_memory_evidence",
    "golden_document_evidence",
    "memory_evidence",
    "evidence_list",
]

csv_path = hf_hub_download(
    HF_DATASET, "experiment_questions.csv", repo_type="dataset"
)
df = pd.read_csv(csv_path)
for col in LIST_COLUMNS:
    df[col] = df[col].map(
        lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
    )

out = Path("data/questions")
out.mkdir(parents=True, exist_ok=True)
df.to_pickle(out / "experiment_210_split.pkl")

snapshot_download(
    repo_id=HF_DATASET,
    repo_type="dataset",
    local_dir="data",
    allow_patterns=[
        "memory_collection/*",
        "document_collection/*",
    ],
)
```

The other artifacts land at the paths the harness expects. Generate a design matrix locally (D-optimal search is stochastic, so this will not match the thesis cell assignments):

```bash
Rscript prepare_data/optimal_design.R --n-questions 198
```


| Artifact              | Path                                                |
| --------------------- | --------------------------------------------------- |
| Design matrix         | `data/experiment_design.csv` (generated, not on Hub) |
| Persona memory        | `data/memory_collection/persona_{1,2,3}.json`       |
| Document corpus       | `data/document_collection/multihop_corpus.jsonl`    |
| Evidence id → URL map | `data/document_collection/evidence_id_to_url.jsonl` |


Then continue at [§5 Agent inference](#5-agent-inference).

## 4. Dataset generation

Skip this section if you only want to evaluate agents; use [§3 Hugging Face dataset](#3-hugging-face-dataset) instead.

Many of the steps can be run more efficiently, but this walkthrough follows the exact order used during the research so the artifacts in `data/` stay easy to map onto the pipeline.

### 4.1 Gathering the sampling frame

1. Split MultiHop-RAG into a 500-question sample and the remainder, and apply the per-question memory/document evidence split. Downstream these two sets (sampled and rest) are treated as one joint set, so the split is ultimately unnecessary — but this is how it was done:

```bash
python -m prepare_data.knowledge_split
```

1. Screen both pools with the golden-context and no-context control agents (writes CSVs that the next step expects). Each command takes one pickle and one output CSV, so sampled and rest are four separate invocations. They can run in parallel:

```bash
python -m sampling_frame_agents.golden_context_agent \
  --questions-pkl data/questions/sampled_questions.pkl \
  --no-author \
  --output-csv data/golden_context_agent_inferences/sampled/llama-3.3-70b-versatile_no_author.csv

python -m sampling_frame_agents.no_context_agent \
  --questions-pkl data/questions/sampled_questions.pkl \
  --output-csv data/no_context_agent_inferences/sampled/llama-3.3-70b-versatile.csv

python -m sampling_frame_agents.golden_context_agent \
  --questions-pkl data/questions/rest_questions.pkl \
  --no-author \
  --output-csv data/golden_context_agent_inferences/rest/llama-3.3-70b-versatile_no_author.csv

python -m sampling_frame_agents.no_context_agent \
  --questions-pkl data/questions/rest_questions.pkl \
  --output-csv data/no_context_agent_inferences/rest/llama-3.3-70b-versatile.csv
```

1. Keep questions that are EM-correct with golden evidence and not EM-correct with no context. Sampled and rest results are joined here:

```bash
python -m prepare_data.collect_reasonable_questions
```

### 4.2 Experiment design

1. Assign personas from evidence domain tags, export the document corpus, sample the 210-question experiment set, and apply the 50-50 integrated split:

```bash
python -m prepare_data.questions_to_personas
python -m prepare_data.prepare_documents
python -m prepare_data.sample_experiment_questions
python -m prepare_data.apply_experiment_split
```

1. Build the D-optimal design matrix (`data/experiment_design.csv`):

```bash
Rscript prepare_data/optimal_design.R
```

### 4.3 Memory generation
1. Submit the experiment evidence-session batch (drop `--no-submit` to upload to the OpenAI Batch API). After the job completes, place `batch_output.jsonl` (and the matching `batch_manifest_*.json`) under `data/batch_jobs/experiment_sessions/`:

```bash
python -m session_generation.create_batch_job \
  --data_path data/questions/experiment_210_split.pkl \
  --output_dir data/batch_jobs/experiment_sessions \
  --no-submit
```

1. Optionally generate a supplement batch from the full reasonable set if the experiment pool is short of 500 sessions per persona:

```bash
python -m session_generation.create_batch_job \
  --data_path data/questions/full_reasonable.pkl \
  --output_dir data/batch_jobs/reasonable_sessions \
  --no-submit
```

1. Inventory pools, then assemble the per-persona 500-session memory corpora. If inventory still shows a shortfall, generate an extra supplement batch and rebuild:

```bash
python -m memory_curation.build_persona_corpus --inventory-only
python -m memory_curation.build_persona_corpus

python -m memory_curation.build_supplement_batch --inventory-only
python -m memory_curation.build_supplement_batch
python -m memory_curation.build_persona_corpus
```

1. Audit generated evidence sessions (optional; used for the session-level analysis):

```bash
python -m session_generation.session_audit.run_audit \
  --memory-dir data/memory_collection \
  --manifest data/batch_jobs/experiment_sessions/batch_manifest.json \
  --out data/session_audit/evidence_field_locations.jsonl
```



## 5. Agent inference

Run one registered agent through its design-matrix rows. Outputs land in `data/experiment_runs/<agent_id>.jsonl`.

```bash
python -m experiment.run_agent --agent agent_1
python -m experiment.run_agent --agent agent_2
python -m experiment.run_agent --agent agent_3
python -m experiment.run_agent --agent agent_4
```


| ID        | Condition      |
| --------- | -------------- |
| `agent_1` | Separate store |
| `agent_2` | MemPalace      |
| `agent_3` | Corrective     |
| `agent_4` | Unified store  |


Use `--dry-run` to exercise the harness without model calls, or `--limit N` for a short smoke run.

## 6. Analysis

The experiment set has 210 questions. Confirmatory analysis drops the 12 `null_query` question blocks (unanswerable items with no golden evidence), so models and tables in `analysis/main_analysis.qmd` use the remaining 198. Those 12 blocks are still present in the design matrix and agent run logs.

Label raw runs (EM correctness and in-domain flags), then export model-feature tables:

```bash
python analysis/collect_evaluation_metadata.py \
  --runs-dir data/experiment_runs \
  --questions-pkl data/questions/experiment_210_split.pkl \
  --output data/experiment_runs/labeled_runs.csv

python analysis/prepare_model_features.py
```

Render the Quarto analyses:

```bash
quarto render analysis/main_analysis.qmd
quarto render analysis/audit_analysis.qmd
```

