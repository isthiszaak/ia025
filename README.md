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
  These are prepared separately; expected answers are not indexed. CARF is not
  included in this baseline.
- `retrieval_data/baseline_irpf.json`: the default index selects three documents:
  Lei nº 9.250/1995, IN RFB nº 2.178, and Lei nº 7.713. This produces 142 chunks.
  Use `--selection all` to build a separate index for all 461 documents.

The initial selection is a small integration baseline with limited subject
coverage. Legal texts reflect the dataset's 2024 source material and can include
historical redactions. Smoke questions have no invented relevance labels.

| Setting | Default |
| --- | --- |
| Embedding model | `embeddinggemma:300m`, 768 dimensions |
| Chunk size / overlap | 1,200 / 200 Python Unicode characters |
| Distance | Cosine distance (`1 - cosine_similarity`); lower is closer |
| Top-k / embedding batch size | 5 / 32 |
| Data / index | `data/retrieval/` / `data/retrieval/chroma/` |

Chunking prefers paragraph, sentence, line, and word boundaries, with hard cuts
when needed. Each chunk retains its source, stable IDs, and `[start_char,
end_char)` offsets in the normalized document. The embedding profile applies
EmbeddingGemma's document/query prefixes. Its 2,048-token input limit is enforced
with Ollama's `truncate:false`; reduce chunk size if an input exceeds it.

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
$env:RETRIEVAL_OLLAMA_BASE_URL = 'http://localhost:11435'

# Download/prepare the full legal corpus and separate question cases.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval prepare

# Build or resume the selected corpus; repeated builds reuse existing chunks.
.\scripts\retrieval\build.ps1
.\scripts\retrieval\query.ps1 'Quem pode ser dependente no IRPF?'
```

If needed, setup accepts `-Python 'C:\path with spaces\python.exe'`. The optional
`.\scripts\retrieval\start-local-ollama.ps1 -OllamaPath 'C:\path\ollama.exe' -PullModel` starts or
reuses a local instance on port 11435. It keeps model files, keys, and logs under
ignored local directories; no Ollama binary or model weights are versioned.

The cache is `data/retrieval/raw/<revision>/`. Prepared artifacts are
`prepared/corpus.jsonl`, `prepared/questions.jsonl`, and `prepared/manifest.json`.
`prepare --raw-dir <directory>` can read the two upstream JSON files locally;
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
.\scripts\retrieval\query.ps1 -Cases retrieval_data/smoke_questions.jsonl -OutputDir retrieval_results
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval query --cases data/retrieval/prepared/questions.jsonl --limit 10 --output-dir retrieval_results
```

Each saved run creates `retrieval_results/<run_id>/retrieval.jsonl` and
`manifest.json`. Rows retain case IDs, inputs, available references,
`retrieved_contexts`, full `retrieved_chunks`, latency, and status/errors.
Original `all_formatted_references` remain available. References are metadata;
only the question is embedded. The manifest records effective configuration,
corpus/model identity, dependency versions, and the input file hash.
Without `--output-dir`, one question prints JSON and cases print JSONL.
The command exits nonzero on errors. Saved retrieval results are available for
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

# Different chunking requires a corresponding index.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval build --chunk-size 800 --chunk-overlap 200
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval query 'Quem pode ser dependente no IRPF?' --chunk-size 800 --chunk-overlap 200

# Optional full legal corpus, stored in a separate collection.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval build --selection all
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval query 'Quem pode ser dependente no IRPF?' --selection all
```

The collection fingerprint includes the corpus hash, selected document IDs,
model digest, dimensions, embedding profile, chunking version/parameters, and
distance. Top-k does not change index identity. Missing, incompatible, or
incomplete indexes cause explicit errors; query never creates an empty index.
Reopening requires the prepared corpus and manifest as well as Chroma storage.
An interrupted build can resume missing chunks. If a hard termination leaves a
`.build.lock`, remove only that lock after confirming no build is active.

Run the 40 retrieval tests with `.\scripts\retrieval\test.ps1`. They cover data preparation,
HTTP failures/validation, persistent Chroma, separate-process reopening,
idempotence, configuration compatibility, and corpus selection. Chroma tests
use a real temporary database with deterministic simulated embeddings; they do
not require Ollama or establish retrieval quality. Real-service checks can be
repeated by building twice and querying in a fresh process. Data, indexes,
models, virtual environments, logs, and generated run records are ignored by Git.

## ⚠️ Troubleshooting

**"Address already in use" Error on Startup**
If you have a native version of Ollama (or an extension like Continue in VS Code) running directly on your host machine, it may block Docker from binding to the required ports.

* **Solution:** Kill the conflicting host processes and try again. The project maps Ollama to port `11435` on the host to minimize these collisions while keeping internal Docker communication on `11434`.

**Chatbot is unresponsive or throwing connection errors**
Ensure the model finished downloading completely. If you just ran `make up` for the first time, Ollama is likely still pulling the weights. You can monitor the download progress and backend activity by running `make logs`.
