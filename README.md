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

## ⚠️ Troubleshooting

**"Address already in use" Error on Startup**
If you have a native version of Ollama (or an extension like Continue in VS Code) running directly on your host machine, it may block Docker from binding to the required ports.

* **Solution:** Kill the conflicting host processes and try again. The project maps Ollama to port `11435` on the host to minimize these collisions while keeping internal Docker communication on `11434`.

**Chatbot is unresponsive or throwing connection errors**
Ensure the model finished downloading completely. If you just ran `make up` for the first time, Ollama is likely still pulling the weights. You can monitor the download progress and backend activity by running `make logs`.
