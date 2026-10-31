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
└── frontend/
    ├── app.py                # Streamlit chat interface
    ├── Dockerfile            # Container definition for the Streamlit app
    └── requirements.txt      # Python dependencies for the frontend

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
* **Pre-requisite Models**: Ensure target models are already pulled (`ollama pull <model-name>`).

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
