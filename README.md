# Local RAG Chatbot

A fully local, privacy-first RAG chatbot built with **Streamlit**, **Ollama**, and **ChromaDB**. The project provides a local LLM interface, document retrieval, and evaluation workflows for measuring generation, retrieval, and end-to-end RAG performance.

The application runs with Docker Compose and supports automatic hardware detection through the Makefile, switching between CPU-only and NVIDIA GPU configurations.

## 📂 Project Structure

```text
.
├── Makefile
├── README.md
├── pyproject.toml
├── docker-compose.cpu.yaml
├── docker-compose.gpu.yaml
│
├── app/
│   ├── Dockerfile
│   ├── streamlit.py                 # Main Streamlit application
│   ├── requirements.txt
│   │
│   ├── backend/
│   │   ├── llm.py                   # LLM configuration and integration
│   │   ├── pipeline.py              # Conversational generation pipeline
│   │   └── prompts.py               # Prompt templates
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── __main__.py              # Retrieval CLI entry point
│   │   ├── config.py                # Retrieval configuration
│   │   ├── data.py                  # Corpus preparation and data loading
│   │   ├── chunking.py              # Document chunking
│   │   ├── tokenizer.py             # Tokenizer integration
│   │   ├── embeddings.py            # Embedding generation
│   │   ├── index.py                 # ChromaDB index management
│   │   ├── experiments.py           # Retrieval experiments
│   │   └── requirements.txt
│   │
│   └── evaluation/
│       ├── runner.py                # Standalone generation evaluation
│       ├── end_to_end_runner.py     # End-to-end RAG evaluation
│       ├── ragas_adapter.py         # RAGAS scoring
│       ├── requirements.txt
│       ├── evaluation_data/
│       │   └── cases.jsonl         # Evaluation dataset
│       └── evaluation_results/      # Generated evaluation runs
│
├── data/
│   └── retrieval/
│       ├── baseline_irpf.json       # Baseline corpus selection
│       ├── smoke_questions.jsonl    # Retrieval smoke-test questions
│       ├── raw/                     # Downloaded source datasets
│       ├── prepared/                # Processed corpus and questions
│       ├── tokenizer/               # Downloaded tokenizer files
│       ├── models/                  # Local model-related artifacts
│       ├── chroma/                  # Persistent ChromaDB indexes
│       └── experiments/             # Retrieval experiment results
│
├── docs/
│
├── scripts/
│   └── retrieval/
│       ├── build.ps1
│       ├── compare.ps1
│       ├── query.ps1
│       ├── setup.ps1
│       ├── start-local-ollama.ps1
│       ├── test.ps1
│       └── validate_tokens.py
│
└── tests/
```

### Main Components

| Directory | Responsibility |
| --- | --- |
| `app/backend/` | LLM integration, conversational pipeline, and prompt management. |
| `app/retrieval/` | Corpus preparation, chunking, embeddings, vector indexing, and retrieval experiments. |
| `app/evaluation/` | Standalone generation evaluation, end-to-end RAG evaluation, and RAGAS scoring. |
| `app/evaluation/evaluation_data/` | Evaluation questions and reference answers. |
| `app/evaluation/evaluation_results/` | Generated predictions, scores, manifests, and evaluation summaries. |
| `data/retrieval/` | Retrieval datasets, processed corpus, tokenizer files, ChromaDB indexes, and experiment outputs. |
| `docs/` | Technical documentation and validation reports. |
| `scripts/retrieval/` | Auxiliary PowerShell and Python scripts for retrieval workflows. |
| `tests/` | Automated tests for retrieval and evaluation components. |

**Generated artifacts:** ChromaDB indexes, prepared datasets, downloaded tokenizer files, and evaluation results are generated locally. They should not be treated as source code or required to be committed to version control unless explicitly needed.

## ⚙️ Prerequisites

- Docker and Docker Compose
- Make
- Python 3.10+ for local development and CLI execution
- NVIDIA Container Toolkit (optional, for GPU acceleration)

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone git@github.com:isthiszaak/ia025.git
cd ia025
```

### 2. Configure environment variables

Create your local environment file:

```bash
cp .env.example .env
```

Configure the models and endpoints according to your environment.

Example:

```env
# Main chatbot
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_BASE_URL=http://ollama:11434

# Retrieval
RETRIEVAL_EMBEDDING_MODEL=embeddinggemma:300m
RETRIEVAL_EMBEDDING_DIMENSION=768
RETRIEVAL_EMBEDDING_PROFILE=embeddinggemma
RETRIEVAL_CHUNK_UNIT=tokens
RETRIEVAL_CHUNK_SIZE=512
RETRIEVAL_CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=5
RETRIEVAL_BATCH_SIZE=32
RETRIEVAL_CORPUS_SELECTION=data/retrieval/baseline_irpf.json

# RAGAS evaluation
EVAL_LLM_MODEL=qwen2.5:1.5b
EVAL_LLM_BASE_URL=http://ollama:11434/v1
EVAL_LLM_TEMPERATURE=0
EVAL_LLM_MAX_TOKENS=2048

EVAL_EMBEDDING_MODEL=nomic-embed-text
EVAL_EMBEDDING_BASE_URL=http://ollama:11434/v1
EVAL_TIMEOUT_SECONDS=600
```

**Important:** The Ollama endpoints depend on where the application runs:

| Execution environment                   | Ollama endpoint          |
| --------------------------------------- | ------------------------ |
| Inside Docker Compose network           | `http://ollama:11434`    |
| Host machine using Compose port mapping | `http://localhost:11435` |
| Native Ollama installation              | `http://localhost:11434` |

The RAGAS endpoints use Ollama's OpenAI-compatible API and therefore include `/v1`. The main chatbot and retrieval configuration use the Ollama root URL, without `/v1`.

The model variables are independent:

- `OLLAMA_MODEL`: chatbot generation model.
- `RETRIEVAL_EMBEDDING_MODEL`: document embedding model.
- `EVAL_LLM_MODEL`: model used by RAGAS.
- `EVAL_EMBEDDING_MODEL`: embedding model used by RAGAS.

Ensure all required models are available in the Ollama instance used by the application.

### 3. Start the application

```bash
make up
```

The Makefile detects available NVIDIA hardware and selects the appropriate Docker Compose configuration.

### 4. Access the application

Open:

```text
http://localhost:8501
```

## 🛠️ Makefile Commands

The Makefile provides shortcuts for managing Docker containers and executing the retrieval pipeline. It also automatically detects NVIDIA GPU availability and selects the appropriate Docker Compose configuration.

### Hardware Detection

Before executing Docker commands, the Makefile checks whether `nvidia-smi` is available:

- **GPU detected:** uses `docker-compose.gpu.yaml`.
- **No GPU detected:** uses `docker-compose.cpu.yaml`.

No manual configuration is required to select the hardware profile.

### Docker Commands

| Command | Description |
| --- | --- |
| `make up` | Starts all containers in detached mode. |
| `make down` | Stops containers and removes associated networks. |
| `make rebuild` | Rebuilds Docker images and starts the containers. |
| `make logs` | Follows logs from all services. |
| `make clean-all` | Stops containers and removes orphans, networks, and volumes. |

**Note:** `make clean-all` also removes persistent Docker volumes, including locally stored models and databases.

### Retrieval Pipeline Commands

The Makefile provides commands to prepare the retrieval corpus, build ChromaDB indexes, and execute retrieval experiments.

| Command | Description |
| --- | --- |
| `make retrieval-prepare` | Prepares the retrieval corpus. |
| `make retrieval-build` | Ensures Ollama is running and the embedding model is available, then builds the ChromaDB index. |
| `make retrieval-setup` | Executes corpus preparation and index building. |
| `make retrieval-compare` | Runs the retrieval comparison experiments using the configured evaluation cases. |

### Retrieval Configuration

The following variables can be overridden directly when running Make commands:

| Variable | Default | Description |
| --- | --- | --- |
| `RETRIEVAL_EMBEDDING_MODEL` | `embeddinggemma:300m` | Ollama embedding model used for retrieval. |
| `RETRIEVAL_SELECTION` | `data/retrieval/baseline_irpf.json` | Corpus selection used to build and evaluate indexes. |
| `RETRIEVAL_CHUNK_SIZE` | `512` | Number of tokens per document chunk. |
| `RETRIEVAL_CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks. |
| `RETRIEVAL_CHUNK_UNIT` | `tokens` | Unit used for document chunking. |
| `RETRIEVAL_TOP_K` | `3` | Number of retrieved chunks per query. |
| `RETRIEVAL_CASES` | `app/evaluation/evaluation_data/cases.jsonl` | Evaluation dataset used in retrieval experiments. |
| `RETRIEVAL_OUTPUT` | `data/retrieval/experiments` | Directory where experiment results are stored. |

### Usage Examples

(MANDATORY) Prepare the corpus and build the baseline index:

```bash
make retrieval-setup
```

Run retrieval experiments with the default configuration:

```bash
make retrieval-compare
```

Override the chunk size and top-k:

```bash
make retrieval-compare \
  RETRIEVAL_CHUNK_SIZE=1024 \
  RETRIEVAL_TOP_K=5
```

Use a different corpus selection:

```bash
make retrieval-build \
  RETRIEVAL_SELECTION=data/retrieval/baseline_irpf.json
```

Change the evaluation dataset and output directory:

```bash
make retrieval-compare \
  RETRIEVAL_CASES=app/evaluation/evaluation_data/cases.jsonl \
  RETRIEVAL_OUTPUT=data/retrieval/experiments
```

### Ollama Model Availability

The `retrieval-build` command automatically starts the Ollama container and checks whether the configured embedding model is available.

It polls Ollama for up to six minutes, checking every two seconds. If the model is not available within that period, the command exits with an error.

This avoids attempting to build the vector index before the embedding service is ready.
---

# 🖥️ Streamlit Application

The application is organized into five tabs, each with a distinct responsibility.

## 1. Chat

The main conversational interface.

- Sends user messages to the locally hosted LLM.
- Maintains conversation history.
- Uses the prompt configured in the Prompt Editor.
- Provides the basic chatbot interaction.

## 2. Prompt Editor

Allows users to customize the chatbot prompt template.

- Edit the prompt used by the Chat tab.
- Save changes for subsequent messages.
- Preserve the required `{user_input}` and {chat_history} placeholders.

The standalone evaluation runner uses its configured prompt template independently of changes made in the Streamlit editor.

## 3. Evaluation Dataset

Manages the dataset and evaluates standalone answer generation.

Main features:

- View, edit, add, and remove evaluation cases.
- Save changes to `app/evaluation/evaluation_data/cases.jsonl`.
- Execute the standalone generation pipeline.
- Select previous generation runs and calculate RAGAS metrics.

This tab evaluates answer generation without document retrieval. Consequently, its generated predictions contain an empty `retrieved_contexts` list.

## 4. Retrieval Experiments

Evaluates document retrieval independently of answer generation.

The experiment uses an **One-Factor-at-a-Time (OFAT)** strategy, producing five unique configurations. For a Chunk size = 512 and Top-k = 3:

| Configuration | Chunk size | Top-k |
| ------------- | ---------: | ----: |
| Chunk 256     |        256 |     3 |
| Chunk 512     |        512 |     3 |
| Chunk 1024    |       1024 |     3 |
| Top-k 1       |        512 |     1 |
| Top-k 5       |        512 |     5 |

The reference configuration is shared between the two comparison groups and is executed only once.

Users can configure:

- Reference chunk size and top-k.
- Chunk overlap.
- Corpus selection: baseline or entire prepared corpus.
- Number of evaluation questions.
- Automatic index building or reuse.

This tab evaluates retrieval behavior, including retrieved chunks, search latency, and execution errors. It does not generate answers or calculate answer-quality metrics.

## 5. End-to-End RAG Evaluation

Evaluates the complete RAG pipeline, combining retrieval, answer generation, and RAGAS evaluation.

The intended workflow is:

1. **Retrieval:** retrieve document chunks for each evaluation question using the selected retrieval configurations.
2. **Generation:** generate an answer conditioned on the question and its retrieved contexts.
3. **Scoring:** evaluate the generated answer and retrieved contexts using RAGAS.
4. **Comparison:** aggregate and compare results across retrieval configurations.

The evaluation considers the following RAGAS metrics:

| Metric             | What it evaluates                                                                     |
| ------------------ | ------------------------------------------------------------------------------------- |
| Answer Relevancy   | Whether the generated answer addresses the user's question.                           |
| Answer Correctness | Whether the answer agrees with the reference answer.                                  |
| Faithfulness       | Whether the answer is supported by the retrieved contexts.                            |
| Context Precision  | Whether the retrieved contexts are relevant to the question.                          |
| Context Recall     | Whether the retrieved contexts contain the information needed to answer the question. |

Metrics requiring reference answers are only applicable to cases that provide a non-empty `reference`.

Unlike the standalone generation evaluation, this workflow uses the retrieved contexts as input to answer generation and evaluation.

---

# 📚 Data Preparation & Retrieval

The `app.retrieval` module provides corpus preparation, local embeddings, token-based chunking, and persistent ChromaDB indexes.

## Dataset

The initial dataset is [unicamp-dl/BR-TaxQA-R](https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R), using the pinned revision:

`9f0f4263928c6152a40f888da3e63375a24bb0f3`

The main source files are:

- `referred_legal_documents_QA_2024_v1.1.json`: 478 records, resulting in 461 unique legal documents.
- `questions_QA_2024_v1.1.json`: 715 questions, answers, and references.
- `acordaos_CARF_2023.json`: 7,204 case-law documents, optionally included with `prepare --include-carf`.

The default baseline selection is:

`data/retrieval/baseline_irpf.json`

It selects three documents:

- Lei nº 9.250/1995
- IN RFB nº 2.178
- Lei nº 7.713

The baseline is intended for initial integration and experimentation. It does not represent the entire legal corpus.

## Retrieval Configuration

| Setting              | Default                  |
| -------------------- | ------------------------ |
| Embedding model      | `embeddinggemma:300m`    |
| Embedding dimension  | 768                      |
| Chunk unit           | Tokens                   |
| Chunk size           | 512                      |
| Chunk overlap        | 50                       |
| Top-k                | 5                        |
| Embedding batch size | 32                       |
| Distance metric      | Cosine                   |
| Data directory       | `data/retrieval/`        |
| Index directory      | `data/retrieval/chroma/` |

Token chunking uses the pinned EmbeddingGemma tokenizer:

`onnx-community/embeddinggemma-300m-ONNX`

Revision:

`8cd82455fa7fae53af154775fd1494f5577ef4b0`

The tokenizer is downloaded separately; embedding generation remains handled by Ollama.

Chunk size and overlap determine index identity. Changing these values requires a corresponding index. Top-k changes the number of retrieved results but does not require a new index.

## Python Interface

```python
from app.retrieval import RetrievalConfig, Retriever

retriever = Retriever(RetrievalConfig.from_env())

results = retriever.retrieve(
    "Quem pode ser dependente no IRPF?",
    top_k=5
)

contexts = [item["text"] for item in results]
sources = [item["source"] for item in results]
```

Each retrieved result includes its text, source, document and chunk identifiers, metadata, distance, and distance metric.

Distance represents vector proximity, not a retrieval quality score or probability.

## Retrieval CLI

Run commands from the repository root.

Prepare the corpus:

```bash
python -m app.retrieval prepare
```

Build or resume the baseline index:

```bash
python -m app.retrieval build --use-prepared
```

Retrieve documents for a question:

```bash
python -m app.retrieval query "Quem pode ser dependente no IRPF?"
```

Run a retrieval comparison:

```bash
python -m app.retrieval compare \
  --cases app/evaluation/evaluation_data/cases.jsonl \
  --output-dir data/retrieval/experiments \
  --chunk-unit tokens \
  --chunk-overlap 50 \
  --chunk-size 512 \
  --top-k 3 \
  --selection data/retrieval/baseline_irpf.json \
  --build-indexes
```

Use `--selection all` to select the entire prepared corpus. Omit `--build-indexes` when compatible indexes already exist.

---

# 📊 Offline Evaluation & RAGAS

The project separates standalone generation evaluation from end-to-end RAG evaluation.

## Evaluation Dataset Format

Evaluation cases are stored as JSONL in:

`app/evaluation/evaluation_data/cases.jsonl`

Example:

```json
{
  "case_id": "example_001",
  "user_input": "Quanto é 2 + 2?",
  "reference": "4"
}
```

Fields:

- `case_id`: optional identifier for the case.
- `user_input`: required non-empty question.
- `reference`: optional expected answer, used by reference-dependent evaluation metrics.

Each line represents one independent evaluation case.

## Standalone Generation

Execute the generation runner:

```bash
python -m app.evaluation.runner
```

The runner processes the evaluation dataset using the configured LLM and prompt template. It does not perform document retrieval or RAGAS scoring.

Each execution creates a unique run directory containing:

- `predictions.jsonl`: generated answers, case IDs, optional references, latency, and execution status.
- `manifest.json`: model, prompt, timestamp, input dataset, and execution statistics.

## RAGAS Scoring

Install evaluation dependencies:

```bash
python -m pip install -r app/evaluation/requirements.txt
```

The project targets **RAGAS 0.4.3**, using its modern metric collection interfaces and explicit LLM and embedding configurations.

Score a standalone generation run:

```bash
python -m app.evaluation.ragas_adapter \
  app/evaluation/evaluation_results/<run_id>/predictions.jsonl
```

A custom output directory can be specified:

```bash
python -m app.evaluation.ragas_adapter \
  app/evaluation/evaluation_results/<run_id>/predictions.jsonl \
  --output-dir app/evaluation/evaluation_results/<run_id>/ragas
```

The RAGAS adapter evaluates the metrics supported by the prediction data. In standalone generation runs, context-dependent metrics are not applicable because no retrieval contexts are available.

For end-to-end evaluation, predictions must contain the retrieved contexts alongside the generated answers.

## Evaluation Artifacts

Evaluation runs retain their results locally.

Typical artifacts include:

| File                | Purpose                                                       |
| ------------------- | ------------------------------------------------------------- |
| `predictions.jsonl` | Per-case model responses and evaluation inputs.               |
| `retrieval.jsonl`   | Per-case retrieval results and retrieved chunks.              |
| `scores.jsonl`      | Per-case RAGAS scores and metric statuses.                    |
| `summary.json`      | Aggregated metric statistics.                                 |
| `manifest.json`     | Run configuration, model information, and execution metadata. |

Metric statuses distinguish successful evaluations, errors, skipped cases, and metrics that are not applicable.

Aggregate statistics include mean, standard deviation, valid count, failure count, not-applicable count, and skipped count. Only successful finite scores contribute to the aggregate values.

---

# ⚠️ Troubleshooting

### Ollama connection errors

Check that the Ollama container is running:

```bash
make logs
```

Ensure the configured model is available and that the application is using the correct endpoint.

Remember:

- Use `http://ollama:11434` from inside Docker Compose.
- Use `http://localhost:11435` from the host when using the Compose port mapping.

### Address already in use

A native Ollama process or another application may already be using the required host port.

Stop the conflicting process or adjust the Compose port mapping.

### Missing embedding or evaluation models

Ensure the models configured in `RETRIEVAL_EMBEDDING_MODEL`, `EVAL_LLM_MODEL`, and `EVAL_EMBEDDING_MODEL` have been pulled into the Ollama instance used by the application.

### Missing or incompatible Chroma indexes

Ensure the index was built using the same corpus, embedding model, chunk size, chunk overlap, and chunking configuration used by the retrieval query.

Use the retrieval build command to create or resume the required indexes.

### RAGAS evaluation failures

Check:

- The evaluator endpoints use the OpenAI-compatible `/v1` URL.
- The evaluator models are available in Ollama.
- The evaluation cases contain valid questions and, when required, references.
- End-to-end predictions contain retrieved contexts for context-dependent metrics.

For detailed error information, inspect `scores.jsonl` and the corresponding execution logs.
