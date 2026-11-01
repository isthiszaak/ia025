# Local RAG Chatbot

A fully local, privacy-first chatbot built with **Streamlit** and **Ollama**. This project sets up a local Large Language Model (LLM) and a frontend chat interface using Docker. It includes automatic hardware detection to seamlessly switch between CPU-only mode and NVIDIA GPU acceleration.

## 📂 Project Structure

```text
.
├── Makefile                  # Automation script for running and managing containers
├── README.md                 # Project documentation
├── .env                      # Environment variables (not included in version control)
├── .env.example              # Example environment variables file
├── docker-compose.cpu.yaml   # Base Docker Compose configuration (CPU-only)
├── docker-compose.gpu.yaml   # GPU overrides for NVIDIA acceleration
├── pyproject.toml            # Python project metadata and dependencies (if applicable)
├── app/
│   ├── app.py                # Streamlit chat interface
│   ├── backend/              # LLM configuration and prompts
│   ├── retrieval/            # Dataset preparation, embeddings, and retrieval
│   ├── Dockerfile            # Container definition for the Streamlit app
│   └── requirements.txt      # Chatbot dependencies
├── retrieval_data/           # Corpus selection and retrieval smoke questions
├── scripts/retrieval/        # PowerShell setup, build, query, and test commands
└── tests/                    # Retrieval tests

```

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed on your host machine:

* **Docker** & **Docker Compose**
* **Make** (Standard on Linux/macOS; available via WSL on Windows)
* *(Optional but Recommended)* **NVIDIA Container Toolkit**: Required if you want to run the models using a dedicated NVIDIA GPU.

## 🚀 Quick Start

**1. Clone the Repository**

```bash
git clone git@github.com:isthiszaak/ia025.git
cd ia025
```

**2. Set Up Environment Variables (Ollama Setup)**
The project uses an environment file to determine which LLM to run. Copy the provided example file:

```bash
cp .env.example .env
```

Open the `.env` file in your editor and define your desired model. For example:

```env
OLLAMA_MODEL=qwen2.5:1.5b
```

*(The Ollama container is configured to read this variable and automatically pull the model when it starts).*

**3. Start the Containers**
The project includes a smart Makefile that automatically detects if you have an NVIDIA GPU available and routes to the correct Docker Compose configuration.

```bash
make up
```

*Note: Once the containers start, Ollama will automatically begin downloading the model specified in your `.env` file. This can take a few minutes depending on your internet connection.*

**4. Access the Chat UI**
Once the model is downloaded, open your browser and navigate to:

* **http://localhost:8501**

## 🛠️ Available Make Commands

You can manage the entire lifecycle of the application using the provided `Makefile`.

| Command | Description |
| --- | --- |
| `make up` | Automatically detects hardware (CPU/GPU) and starts the Streamlit and Ollama containers in the background. |
| `make down` | Stops and removes the containers and networks safely. |
| `make rebuild` | Stops containers, removes orphans, forces a rebuild of the Docker images, and restarts the environment. |
| `make logs` | Tails the live logs for both Streamlit and Ollama services. |
| `make clean-all` | **Deep Clean:** Stops all containers and completely removes Docker volumes (deletes downloaded models and databases). |

## Data Preparation & Retrieval

`app.retrieval` provides dataset preparation, local Ollama embeddings, and a
persistent Chroma index. The Python interface and CLI expose retrieved text
and source metadata. The Streamlit chatbot requires a separate integration to
consume these contexts.

### Dataset and initial corpus

The source is [unicamp-dl/BR-TaxQA-R](https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R),
v1.1 JSON files at revision `9f0f4263928c6152a40f888da3e63375a24bb0f3`,
licensed under [CC BY 4.0](https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R/blob/main/LICENSE).
The loader verifies the pinned checksums and preserves provenance in a local
manifest. It normalizes Unicode/line endings and removes exact normalized-text
duplicates while retaining all source filenames as aliases.

- `referred_legal_documents_QA_2024_v1.1.json`: 478 records, yielding 461 unique
  legal documents and 17 duplicate records.
- `questions_QA_2024_v1.1.json`: 715 questions, answers, and original references.
  These are prepared separately; expected answers are not indexed.
- `acordaos_CARF_2023.json`: 7,204 case-law documents, verified in the same
  pinned revision. Opt in with `prepare --include-carf`; the combined corpus has
  7,665 unique documents. This file has no QA pairs, and the 715 questions have
  no explicit CARF file references. No CARF relevance labels are inferred.
- `data/retrieval/baseline_irpf.json`: the default index selects three documents:
  Lei nº 9.250/1995, IN RFB nº 2.178, and Lei nº 7.713. The historical character
  configuration (1,200/200) produced 142 chunks; token configurations differ.
  Use `--selection all` to build a separate index for all 461 documents.

The initial selection is a small integration baseline with limited subject
coverage. Legal texts reflect the dataset's 2024 source material and can include
historical redactions. Smoke questions have no invented relevance labels.

| Setting | Default |
| --- | --- |
| Embedding model | `embeddinggemma:300m`, 768 dimensions |
| CLI chunk size / overlap | 512 / 50 tokens; planned sizes: 256, 512, 1024 |
| Tokenizer | EmbeddingGemma tokenizer, `tokenizers==0.22.2`, pinned revision below |
| Distance | Cosine distance (`1 - cosine_similarity`); lower is closer |
| Top-k / embedding batch size | 5 / 32 |
| Data / index | `data/retrieval/` / `data/retrieval/chroma/` |

Token chunking uses the actual EmbeddingGemma vocabulary from
[`onnx-community/embeddinggemma-300m-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/tree/8cd82455fa7fae53af154775fd1494f5577ef4b0),
revision `8cd82455fa7fae53af154775fd1494f5577ef4b0`, with `tokenizer.json` SHA256
`4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47`.
Only the tokenizer is downloaded, not ONNX model weights; embeddings remain in
Ollama. Token counts are never estimated from character counts.

Chunks are exact slices of the normalized document with stable IDs, source
aliases, character offsets, original-document token offsets and independently
checked token counts. Unicode byte-fallback boundaries retain complete
characters; rare overlap adjustments are recorded as `actual_overlap_tokens`.
The configurable 50-token overlap is an implementation choice and stays constant
across comparisons, always below 256. Prefixes, titles and special tokens do not
count toward chunk size but do count toward the 2,048-token embedding input
limit. Oversized complete inputs fail before HTTP; Ollama also receives
[`truncate:false`](https://docs.ollama.com/api/embed). No text is silently cut.

### Setup and commands on Windows

Run from the repository root in PowerShell. Setup requires Python 3.10+ and
installs `app/retrieval/requirements.txt` into `.venv-retrieval`. It preserves an
existing `.env`; new options are documented in `.env.example`.

Ollama must be running with `embeddinggemma:300m` available (Ollama >= 0.11.10).
For this project's Compose instance, the host URL is `http://localhost:11435`;
inside its network use `http://ollama:11434`. Native Ollama normally uses
`http://localhost:11434`. Use the root URL without `/v1`.
For Compose, pull the model with
`docker exec ia025_ollama ollama pull embeddinggemma:300m`. A native CLI can use
`ollama pull embeddinggemma:300m` with `OLLAMA_HOST` set to the same instance.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\retrieval\setup.ps1
$env:OLLAMA_BASE_URL = 'http://localhost:11435'

# Download/prepare the full legal corpus and separate question cases.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval prepare

# Build or resume the selected corpus; repeated builds reuse existing chunks.
.\scripts\retrieval\build.ps1
.\scripts\retrieval\query.ps1 'Quem pode ser dependente no IRPF?'

# Prepare the legal + CARF corpus separately, preserving the small legal setup.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval prepare --include-carf --data-dir data/retrieval-carf --raw-dir data/retrieval/raw/9f0f4263928c6152a40f888da3e63375a24bb0f3
.\scripts\retrieval\build.ps1 -UsePrepared -DataDir data/retrieval-carf -IndexDir data/retrieval-carf/chroma -Selection all
.\scripts\retrieval\query.ps1 'Quem pode ser dependente no IRPF?' -DataDir data/retrieval-carf -IndexDir data/retrieval-carf/chroma -Selection all
```

If needed, setup accepts `-Python 'C:\path with spaces\python.exe'`. The optional
`.\scripts\retrieval\start-local-ollama.ps1 -OllamaPath 'C:\path\ollama.exe' -PullModel` starts or
reuses a local instance on port 11435. It keeps model files, keys, and logs under
ignored local directories; no Ollama binary or model weights are versioned.

The cache is `data/retrieval/raw/<revision>/`. Prepared artifacts are
`prepared/corpus.jsonl`, `prepared/questions.jsonl`, and `prepared/manifest.json`.
`prepare --raw-dir <directory>` can read the upstream JSON files locally;
`--revision` requires a pinned 40-character Git SHA. Empty/invalid records and
exact duplicates are reported in the preparation manifest.

### Python interface and saved results

```python
from app.retrieval import RetrievalConfig, Retriever

retriever = Retriever(RetrievalConfig.from_env())
results = retriever.retrieve("Quem pode ser dependente no IRPF?", top_k=5)
contexts = [item["text"] for item in results]
sources = [item["source"] for item in results]
```

Each result contains `chunk_id`, `document_id`, `text`, `source`, `title`,
`metadata`, `distance`, and `distance_metric`. Metadata includes source aliases,
dataset revision, and chunk offsets. Questions must be nonempty strings; top-k
must be a positive integer. Distance is not a probability or quality score.

```powershell
.\scripts\retrieval\query.ps1 -Cases data/retrieval/smoke_questions.jsonl -OutputDir retrieval_results
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval query --cases data/retrieval/prepared/questions.jsonl --limit 10 --output-dir retrieval_results
```

Each saved run creates `retrieval_results/<run_id>/retrieval.jsonl` and
`manifest.json`. Rows retain case IDs, inputs, available references,
`retrieved_contexts`, full `retrieved_chunks`, latency, and status/errors.
Original `all_formatted_references` remain available. References are metadata;
only the question is embedded. The manifest records effective configuration,
corpus/model identity, dependency versions, and the input file hash.
Without `--output-dir`, one question prints JSON and cases print JSONL.
Index-opening failures are also saved, with corpus/configuration, error stage,
and null search latency when no search occurred. The command exits nonzero on errors. Saved retrieval results are available for
downstream answer generation.

### Configuration and index reuse

Retrieval reads `RETRIEVAL_*` settings independently of the chatbot's model
configuration. Environment variables override `.env`; CLI options override
retrieval settings. Relative paths from `.env` resolve against the repository
root. Chunk size/overlap, top-k, batch size, model, endpoint, and corpus selection
are configurable without editing code.

```powershell
# Same index, different number of results.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval query 'Quem pode ser dependente no IRPF?' --top-k 3

# Different token chunking requires a corresponding index.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval build --use-prepared --chunk-unit tokens --chunk-size 256 --chunk-overlap 50
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval query 'Quem pode ser dependente no IRPF?' --chunk-unit tokens --chunk-size 256 --chunk-overlap 50

# Optional full legal corpus, stored in a separate collection.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval build --use-prepared --selection all
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval query 'Quem pode ser dependente no IRPF?' --selection all
```

The collection fingerprint includes the corpus hash, selected document IDs,
model digest, dimensions, embedding profile, chunking unit/version/parameters,
tokenizer identifier/revision/file hash/library version (in token mode), and
distance. Top-k does not change index identity. Missing, incompatible, or
incomplete indexes cause explicit errors; query never creates an empty index.
Reopening requires the prepared corpus and manifest as well as Chroma storage.
An interrupted build can resume missing chunks. If a hard termination leaves a
`.build.lock`, remove only that lock after confirming no build is active.

For compatibility, `RetrievalConfig()` and `chunk_documents(...)` keep their
original character defaults. Use `RetrievalConfig.from_env()` for the new token
defaults, or set `chunk_unit="tokens", chunk_size=512, chunk_overlap=50` explicitly.
An older `.env` with size/overlap but no unit retains character interpretation;
set `RETRIEVAL_CHUNK_UNIT=tokens` and update both numbers when migrating.
To reproduce the old baseline explicitly, pass `--chunk-unit characters
--chunk-size 1200 --chunk-overlap 200` to both build and query.
`build --use-prepared` consumes the prepared corpus without downloading or
replacing it; without this flag build retains its original prepare-then-build
behavior, with CARF included only when `--include-carf` is explicit.

Run retrieval tests with `.\scripts\retrieval\test.ps1`. They cover data preparation,
HTTP failures/validation, persistent Chroma, separate-process reopening,
idempotence, configuration compatibility, and corpus selection. Chroma tests
use a real temporary database with deterministic simulated embeddings; they do
not require Ollama or establish retrieval quality. Real-service checks can be
repeated by building twice and querying in a fresh process. Data, indexes,
models, virtual environments, logs, and generated run records are ignored by Git.

### Planned retrieval comparisons and handoff

Prepare the chosen corpus first. The following builds/resumes three token
indexes and runs five unique settings: `(256,3)`, `(512,3)`, `(1024,3)`, `(512,1)`,
`(512,5)`. Chunk size varies at top-k 3; top-k varies at size 512. Overlap, corpus,
tokenizer and embedding model stay fixed. Omitting `-BuildIndexes` reuses indexes.

```powershell
.\scripts\retrieval\compare.ps1 -Cases data/retrieval/prepared/questions.jsonl -Limit 10 -OutputDir retrieval_results -Selection all -ChunkOverlap 50 -BuildIndexes
# Equivalent CLI; omit --limit to use all 715 questions.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval compare --cases data/retrieval/prepared/questions.jsonl --limit 10 --output-dir retrieval_results --selection all --chunk-unit tokens --chunk-overlap 50 --build-indexes
# Offline tokenizer/content/input-limit audit; does not require Ollama.
& '.\.venv-retrieval\Scripts\python.exe' scripts/retrieval/validate_tokens.py --data-dir data/retrieval --report data/retrieval/token_validation.json
```

Comparison manifests and JSONL retain effective configuration, corpus/model
identity, tokenizer, case IDs, original references, chunks, sources, distances,
search latency and failures. No relevance metrics or answer-quality scores are
computed. Distances describe vector proximity only. The Python interface above
and saved JSONL are the handoff to the generation/evaluation owners; these
changes do not connect to their LLM, prompts, evaluator or UI.

Verified counts and source schemas are in [data validation](docs/data-validation.md),
token budgets in [token validation](docs/token-validation.md), and execution
results/limitations in [Member 1 validation](docs/member1-validation.md).
The [second-dataset analysis](docs/second-dataset.md) recommends LARI for a
different domain, or JurisTCU when explicit retrieval relevance judgments are
the priority. Selection remains with the group; no second-dataset ingestion was
implemented.

## ⚠️ Troubleshooting

---

# Offline Evaluation & QA

Generation and scoring are independent offline stages. Generation does not run RAGAS or retrieve documents (`retrieved_contexts` is always an empty list).

---

## Step 1: Answer Generation

### 1. Input Dataset Format

Cases are read line-by-line from `evaluation_data/cases.jsonl`:

```json
{"case_id": "example_001", "user_input": "Quanto é 2 + 2?", "reference": "4"}
```

* **`user_input`**: Required non-empty string.
* **`case_id`**: Optional (defaults to `line_1`, `line_2`, etc.).
* **`reference`**: Optional ground-truth answer (passed through to predictions metadata; ignored by the generator).
* **Handling**: Blank lines are skipped. Invalid cases or generation failures emit an error record and proceed to the next item. Missing input files cause startup errors. The bundled cases serve as smoke examples rather than a benchmark.

---

### 2. Prerequisites & Environment Setup

Ensure Ollama is running with the configured model available.

* **Dependencies**: `app/requirements.txt`
* **Execution Location**: Repository root

> **Note**: The runner uses the standard `CHAT_TEMPLATE` and `get_llm()` config. Edits made in the Streamlit prompt editor do not carry over to runner executions.

---

### 3. Execution Commands

**Standard Execution:**

```bash
python -m app.evaluation.runner
```

**Execution against Docker Compose Ollama (Host Port):**

```bash
OLLAMA_BASE_URL=http://localhost:11435 python -m app.evaluation.runner
```

**Custom Input / Output Paths:**

```bash
python -m app.evaluation.runner --cases evaluation_data/cases.jsonl --output-dir evaluation_results
```

---

### 4. Output Artifacts

Each execution creates a unique directory at `evaluation_results/<run_id>/` (ignored by Git):

* **`predictions.jsonl`**: Contains case ID, line number, user question, model response, optional reference, empty contexts, latency (s), and status. Error records contain `error_type` and `error` with a `null` response.
* **`manifest.json`**: Captures run ID, UTC timestamp, target model, temperature, effective prompt template, input path, and execution counts (total, successful, failed).

---

## Step 2: Scoring Predictions with RAGAS

### 1. Prerequisites & Environment Setup

#### Install Evaluation Dependencies

Install the optional evaluation dependencies (separate from the main chatbot requirements):

```bash
python -m pip install -r app/evaluation/requirements.txt
```

> **Version Pinning Note**: Targets RAGAS **0.4.3** using modern `ragas.metrics.collections` and explicit LLM/embedding factories. Pins `langchain-community==0.4.1` to preserve Vertex AI dependencies required by RAGAS 0.4.3.

#### Environment Configuration (`.env`)

Configure `EVAL_LLM_MODEL` and `EVAL_EMBEDDING_MODEL` explicitly (these do **not** inherit `OLLAMA_MODEL`, check the `.env.example` file for details):

* **API Compatibility**: Both evaluator endpoints use OpenAI-compatible `/v1` endpoints (e.g., Ollama's `/v1`).
* **Port Mapping**: Local Ollama typically uses port `11434`; Compose host port uses `11435`.
* **Requirements**: Evaluator LLM must support structured JSON output; embedding endpoint must support the target model.
* **Pre-requisite Models**: Ensure target models are already pulled (`ollama pull <model-name>`). If you have a Ollama running container, you can use `docker exec <container-id> ollama pull <model-name>` to pull models.

---

### 2. Evaluated Metrics Summary

| Metric | Status | Requirements & Behavior |
| --- | --- | --- |
| **Answer Relevancy** | **Active** | Measures response relevance using evaluator LLM + embeddings. Does not require contexts. |
| **Answer Correctness** | **Active** (Conditional) | Runs only when prediction has a non-empty `reference`. Uses default weights (`0.75` factuality / `0.25` similarity). |
| **Faithfulness** | *Disabled* | Disabled in this offline stage. |
| **Context Precision** | *Disabled* | Disabled in this offline stage. |
| **Context Recall** | *Disabled* | Disabled in this offline stage. |

---

### 3. Execution Commands

**Standard Scoring:**

```bash
python -m app.evaluation.ragas_adapter evaluation_results/<run_id>/predictions.jsonl
```

**Re-scoring to a Custom Directory (Prevents Overwriting):**

```bash
python -m app.evaluation.ragas_adapter evaluation_results/<run_id>/predictions.jsonl --output-dir evaluation_results/<run_id>/another_evaluation
```

---

### 4. Output Artifacts & Error Handling

Scoring outputs are written alongside the input predictions (or into the designated output directory):

* **`scores.jsonl`**: Detailed per-case scores containing `scores`, `metric_status`, and reasons for unavailable metrics:
* *Missing references / Disabled metrics*: Rendered as `null` with status `not_applicable`.
* *Failed generation records*: Rendered with status `skipped`.
* *Metric errors/timeouts*: Rendered as `null` with status `error`. A single failure does not halt processing for other metrics or cases.


* **`summary.json`**: Aggregate statistics (`mean`, population `stddev`, `valid_count`, `failure_count`, `not_applicable_count`, `skipped_count`).
* Only successful finite scores enter statistics.
* 0 valid scores $\rightarrow$ `mean` and `stddev` are `null`.
* 1 valid score $\rightarrow$ `stddev` is `0`.


* **`manifest.json` (Updated)**: Appends an `evaluation` section storing model/embedding config, RAGAS version, metric settings, and timestamp (API keys are omitted).

---

### 5. Running Adapter Tests

Run unit tests without contacting an Ollama server (uses mocked HTTP responses when RAGAS is installed):

```bash
python -m unittest discover -s tests -v
```
## ⚠️ General Troubleshooting

---

# Offline Evaluation & QA

Generation and scoring are independent offline stages. Generation does not run RAGAS or retrieve documents (`retrieved_contexts` is always an empty list).

---

## Step 1: Answer Generation

### 1. Input Dataset Format

Cases are read line-by-line from `evaluation_data/cases.jsonl`:

```json
{"case_id": "example_001", "user_input": "Quanto é 2 + 2?", "reference": "4"}
```

* **`user_input`**: Required non-empty string.
* **`case_id`**: Optional (defaults to `line_1`, `line_2`, etc.).
* **`reference`**: Optional ground-truth answer (passed through to predictions metadata; ignored by the generator).
* **Handling**: Blank lines are skipped. Invalid cases or generation failures emit an error record and proceed to the next item. Missing input files cause startup errors. The bundled cases serve as smoke examples rather than a benchmark.

---

### 2. Prerequisites & Environment Setup

Ensure Ollama is running with the configured model available.

* **Dependencies**: `app/requirements.txt`
* **Execution Location**: Repository root

> **Note**: The runner uses the standard `CHAT_TEMPLATE` and `get_llm()` config. Edits made in the Streamlit prompt editor do not carry over to runner executions.

---

### 3. Execution Commands

**Standard Execution:**

```bash
python -m app.evaluation.runner
```

**Execution against Docker Compose Ollama (Host Port):**

```bash
OLLAMA_BASE_URL=http://localhost:11435 python -m app.evaluation.runner
```

**Custom Input / Output Paths:**

```bash
python -m app.evaluation.runner --cases evaluation_data/cases.jsonl --output-dir evaluation_results
```

---

### 4. Output Artifacts

Each execution creates a unique directory at `evaluation_results/<run_id>/` (ignored by Git):

* **`predictions.jsonl`**: Contains case ID, line number, user question, model response, optional reference, empty contexts, latency (s), and status. Error records contain `error_type` and `error` with a `null` response.
* **`manifest.json`**: Captures run ID, UTC timestamp, target model, temperature, effective prompt template, input path, and execution counts (total, successful, failed).

---

## Step 2: Scoring Predictions with RAGAS

### 1. Prerequisites & Environment Setup

#### Install Evaluation Dependencies

Install the optional evaluation dependencies (separate from the main chatbot requirements):

```bash
python -m pip install -r app/evaluation/requirements.txt
```

> **Version Pinning Note**: Targets RAGAS **0.4.3** using modern `ragas.metrics.collections` and explicit LLM/embedding factories. Pins `langchain-community==0.4.1` to preserve Vertex AI dependencies required by RAGAS 0.4.3.

#### Environment Configuration (`.env`)

Configure `EVAL_LLM_MODEL` and `EVAL_EMBEDDING_MODEL` explicitly (these do **not** inherit `OLLAMA_MODEL`, check the `.env.example` file for details):

* **API Compatibility**: Both evaluator endpoints use OpenAI-compatible `/v1` endpoints (e.g., Ollama's `/v1`).
* **Port Mapping**: Local Ollama typically uses port `11434`; Compose host port uses `11435`.
* **Requirements**: Evaluator LLM must support structured JSON output; embedding endpoint must support the target model.
* **Pre-requisite Models**: Ensure target models are already pulled (`ollama pull <model-name>`). If you have a Ollama running container, you can use `docker exec <container-id> ollama pull <model-name>` to pull models.

---

### 2. Evaluated Metrics Summary

| Metric | Status | Requirements & Behavior |
| --- | --- | --- |
| **Answer Relevancy** | **Active** | Measures response relevance using evaluator LLM + embeddings. Does not require contexts. |
| **Answer Correctness** | **Active** (Conditional) | Runs only when prediction has a non-empty `reference`. Uses default weights (`0.75` factuality / `0.25` similarity). |
| **Faithfulness** | *Disabled* | Disabled in this offline stage. |
| **Context Precision** | *Disabled* | Disabled in this offline stage. |
| **Context Recall** | *Disabled* | Disabled in this offline stage. |

---

### 3. Execution Commands

**Standard Scoring:**

```bash
python -m app.evaluation.ragas_adapter evaluation_results/<run_id>/predictions.jsonl
```

**Re-scoring to a Custom Directory (Prevents Overwriting):**

```bash
python -m app.evaluation.ragas_adapter evaluation_results/<run_id>/predictions.jsonl --output-dir evaluation_results/<run_id>/another_evaluation
```

---

### 4. Output Artifacts & Error Handling

Scoring outputs are written alongside the input predictions (or into the designated output directory):

* **`scores.jsonl`**: Detailed per-case scores containing `scores`, `metric_status`, and reasons for unavailable metrics:
* *Missing references / Disabled metrics*: Rendered as `null` with status `not_applicable`.
* *Failed generation records*: Rendered with status `skipped`.
* *Metric errors/timeouts*: Rendered as `null` with status `error`. A single failure does not halt processing for other metrics or cases.


* **`summary.json`**: Aggregate statistics (`mean`, population `stddev`, `valid_count`, `failure_count`, `not_applicable_count`, `skipped_count`).
* Only successful finite scores enter statistics.
* 0 valid scores $\rightarrow$ `mean` and `stddev` are `null`.
* 1 valid score $\rightarrow$ `stddev` is `0`.


* **`manifest.json` (Updated)**: Appends an `evaluation` section storing model/embedding config, RAGAS version, metric settings, and timestamp (API keys are omitted).

---

### 5. Running Adapter Tests

Run unit tests without contacting an Ollama server (uses mocked HTTP responses when RAGAS is installed):

```bash
python -m unittest discover -s tests -v
```
## ⚠️ General Troubleshooting

**"Address already in use" Error on Startup**
If you have a native version of Ollama (or an extension like Continue in VS Code) running directly on your host machine, it may block Docker from binding to the required ports.

* **Solution:** Kill the conflicting host processes and try again. The project maps Ollama to port `11435` on the host to minimize these collisions while keeping internal Docker communication on `11434`.

**Chatbot is unresponsive or throwing connection errors**
Ensure the model finished downloading completely. If you just ran `make up` for the first time, Ollama is likely still pulling the weights. You can monitor the download progress and backend activity by running `make logs`.
