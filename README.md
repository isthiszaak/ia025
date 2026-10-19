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

## Offline Evaluation & QA

Generation and scoring are independent offline stages. Generation does not run
RAGAS or retrieve documents; `retrieved_contexts` is always an empty list.

From the repository root, in a Python environment with the dependencies in
`app/requirements.txt` installed, run:

```bash
python -m app.evaluation.runner
```

Ollama must be running with the configured model available. The runner uses the
same `get_llm()` configuration and generation function as the chatbot, without
importing Streamlit. It uses the default `CHAT_TEMPLATE`; edits made in the chat
prompt editor apply only to that Streamlit session. For an Ollama instance started
by this project's Docker Compose, use its host port when running the runner on
the host:

```bash
OLLAMA_BASE_URL=http://localhost:11435 python -m app.evaluation.runner
```

Cases are read from `evaluation_data/cases.jsonl`, one JSON object per line:

```json
{"case_id": "example_001", "user_input": "Quanto é 2 + 2?"}
```

`user_input` must be a nonempty string. `case_id` is optional and defaults to the
line number (`line_1`, etc.). The bundled cases are smoke examples, not a quality
benchmark. Blank lines are ignored. Invalid cases and generation failures produce
error records, and processing continues with the next line. Startup failures such
as a missing input file are reported as command errors.

Each execution creates `evaluation_results/<run_id>/` containing:

- `predictions.jsonl`: case ID, input line number, question, response, empty
  contexts, latency in seconds and status. Error records include `error_type`
  and `error`, with a null response.
- `manifest.json`: run ID, UTC timestamp, model, temperature, effective prompt
  template, input path, total cases, successful cases and cases with errors.

Generated results are ignored by Git. To use other input or output paths:

```bash
python -m app.evaluation.runner --cases evaluation_data/cases.jsonl --output-dir evaluation_results
```

### Score existing predictions with RAGAS

Install the optional evaluation dependencies in your Python environment:

```bash
python -m pip install -r app/evaluation/requirements.txt
```

The adapter targets RAGAS **0.4.3**, using `ragas.metrics.collections` and modern
LLM/embedding factories rather than legacy LangChain wrappers. These dependencies
pin `langchain-community==0.4.1` because `0.4.2` removes the Vertex AI module
imported by RAGAS 0.4.3, even when using another model provider. They
are separate from the chatbot's requirements; the chatbot Docker image does not
install RAGAS by default.

Configure `EVAL_LLM_MODEL` and `EVAL_EMBEDDING_MODEL` explicitly in `.env` or the
environment. They do not inherit `OLLAMA_MODEL`. See `.env.example` for all
settings. Both evaluator endpoints use the OpenAI-compatible API (including
Ollama's `/v1` endpoint), with separate URLs and optional API keys. A local Ollama
instance normally uses port `11434`; the Compose instance exposes port `11435`
on the host. The evaluator LLM must support structured JSON output, and the
embedding endpoint must support the selected embedding model. Models must already
be available; this command does not download them.

Run scoring against the predictions you already generated:

```bash
python -m app.evaluation.ragas_adapter evaluation_results/<run_id>/predictions.jsonl
```

This command never regenerates chatbot responses. It writes `scores.jsonl` and
`summary.json` beside the predictions, and adds an `evaluation` section to
`manifest.json`, preserving generation metadata. That section records model and
embedding configuration, RAGAS version, API, metric settings and UTC timestamp;
API keys are not recorded. Existing scores are protected from overwriting. To
evaluate the same predictions again, select another output directory:

```bash
python -m app.evaluation.ragas_adapter evaluation_results/<run_id>/predictions.jsonl --output-dir evaluation_results/<run_id>/another_evaluation
```

The separate output directory receives its own scores, summary and a manifest
carrying the original generation metadata plus the new evaluator configuration.

- **Answer Relevancy** measures response relevance to the question using the
  evaluator LLM and embeddings, without requiring retrieved contexts.
- **Answer Correctness** runs only when the prediction contains a nonempty string
  `reference`, using RAGAS's default factuality/similarity weights of `0.75/0.25`.
  Add an optional `reference` field to dataset cases; the generation runner now
  copies it to predictions as metadata and does not send it to the generator.
  Previously saved predictions without this field remain evaluable for relevancy.
- **Faithfulness, Context Precision and Context Recall** are disabled in this
  stage, including when an input happens to contain contexts.

Every score row identifies the prediction's line and case ID, and contains
`scores`, `metric_status` and reasons for unavailable metrics. Missing references
and disabled metrics are `null` with `not_applicable` status. Failed generation
records are `skipped`. Metric exceptions, timeouts and nonfinite results are
`null` with `error` status; one metric failing does not prevent the other metric
or subsequent cases from running. Malformed prediction lines are also recorded.
Configuration, missing-file and output filesystem errors are command failures.

`summary.json` reports `mean`, population `stddev`, `valid_count`, `failure_count`,
`not_applicable_count` and `skipped_count` for every metric. Only successful finite
scores enter the statistics. With no valid scores, mean and deviation are `null`;
with one valid score, the population deviation is `0`. A genuine zero score is
retained as a valid observation. Relevancy does not measure factual accuracy, and
these scores are not a substitute for human review.

Run adapter tests without requiring an Ollama server:

```bash
python -m unittest discover -s tests -v
```

When RAGAS is installed, the suite also exercises the actual metric factories and
collections API against mocked HTTP responses, without contacting a model server.

## ⚠️ Troubleshooting

**"Address already in use" Error on Startup**
If you have a native version of Ollama (or an extension like Continue in VS Code) running directly on your host machine, it may block Docker from binding to the required ports.

* **Solution:** Kill the conflicting host processes and try again. The project maps Ollama to port `11435` on the host to minimize these collisions while keeping internal Docker communication on `11434`.

**Chatbot is unresponsive or throwing connection errors**
Ensure the model finished downloading completely. If you just ran `make up` for the first time, Ollama is likely still pulling the weights. You can monitor the download progress and backend activity by running `make logs`.
