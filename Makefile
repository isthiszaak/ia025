.PHONY: help up down rebuild logs clean-all \
        retrieval-prepare retrieval-build retrieval-setup retrieval-compare

# ============================================================
# CONFIGURATION
# ============================================================

PYTHON ?= python

# Retrieval configuration
RETRIEVAL_EMBEDDING_MODEL ?= embeddinggemma:300m
RETRIEVAL_SELECTION ?= data/retrieval/baseline_irpf.json
RETRIEVAL_CHUNK_SIZE ?= 512
RETRIEVAL_CHUNK_OVERLAP ?= 50
RETRIEVAL_CHUNK_UNIT ?= tokens
RETRIEVAL_TOP_K ?= 3

RETRIEVAL_CASES ?= app/evaluation/evaluation_data/cases.jsonl
RETRIEVAL_OUTPUT ?= data/retrieval/experiments

# Detect if an NVIDIA GPU is available on the host machine
HAS_GPU := $(shell nvidia-smi > /dev/null 2>&1 && echo yes || echo no)

# Defines docker compose flags mutually exclusively
ifeq ($(HAS_GPU),yes)
    COMPOSE_FLAGS := -f docker-compose.gpu.yaml
else
    COMPOSE_FLAGS := -f docker-compose.cpu.yaml
endif


# ============================================================
# HELP
# ============================================================

help:
	@echo "Available commands:"
	@echo ""
	@echo "Docker:"
	@echo "  make up                 Start containers"
	@echo "  make down               Stop containers"
	@echo "  make rebuild            Rebuild and start containers"
	@echo "  make logs               Follow container logs"
	@echo "  make clean-all          Remove containers and volumes"
	@echo ""
	@echo "Retrieval:"
	@echo "  make retrieval-prepare  Prepare retrieval corpus"
	@echo "  make retrieval-build   Build Chroma vector index"
	@echo "  make retrieval-setup   Prepare corpus and build Chroma index"
	@echo "  make retrieval-compare Run retrieval experiments"
	@echo ""
	@echo "Optional retrieval parameters:"
	@echo "  RETRIEVAL_SELECTION, RETRIEVAL_CHUNK_SIZE,"
	@echo "  RETRIEVAL_CHUNK_OVERLAP, RETRIEVAL_TOP_K"


# ============================================================
# DOCKER
# ============================================================

# Brings up the containers in the background
up:
	@if [ "$(HAS_GPU)" = "yes" ]; then \
		echo "NVIDIA GPU detected! Starting containers with GPU acceleration..."; \
	else \
		echo "No NVIDIA GPU detected. Starting containers in CPU mode..."; \
	fi
	docker compose $(COMPOSE_FLAGS) up -d
	@echo "All containers have been started."
	@echo "Check if Ollama is pulling the model. This may take a few minutes."

# Brings down the containers and explicitly removes orphans
down:
	docker compose $(COMPOSE_FLAGS) down
	@echo "All containers have been stopped."

# Rebuilds all containers and starts them
rebuild:
	@echo "Rebuilding all containers..."
	@if [ "$(HAS_GPU)" = "yes" ]; then \
		echo "NVIDIA GPU detected! Starting containers with GPU acceleration..."; \
	else \
		echo "No NVIDIA GPU detected. Starting containers in CPU mode..."; \
	fi
	docker compose $(COMPOSE_FLAGS) up -d --build
	@echo "All containers have been rebuilt and started."
	@echo "Wait for Ollama to finish pulling the model before using the frontend."

# Tails the logs for all services
logs:
	docker compose $(COMPOSE_FLAGS) logs -f

# Deep clean: stops containers, removes orphans, and deletes volumes
clean-all:
	docker compose $(COMPOSE_FLAGS) down --remove-orphans --volumes
	@echo "Deep clean complete. All containers, networks, and volumes have been removed."


# ============================================================
# RETRIEVAL PIPELINE
# ============================================================

# Step 1: Prepare corpus and retrieval data
retrieval-prepare:
	@echo "Preparing retrieval corpus..."
	$(PYTHON) -m app.retrieval prepare
	@echo "Retrieval corpus preparation completed."

# Step 2: Build Chroma vector index
retrieval-build: retrieval-ollama
	@echo "Building Chroma vector index..."
	$(PYTHON) -m app.retrieval build \
		--selection $(RETRIEVAL_SELECTION) \
		--chunk-size $(RETRIEVAL_CHUNK_SIZE) \
		--chunk-overlap $(RETRIEVAL_CHUNK_OVERLAP) \
		--chunk-unit $(RETRIEVAL_CHUNK_UNIT)
	@echo "Chroma index build completed."
	
# Complete preparation: corpus + Chroma index
retrieval-setup: retrieval-prepare retrieval-build
	@echo ""
	@echo "Retrieval environment is ready!"
	@echo "Corpus prepared and Chroma index built."

# Run retrieval comparison experiments
retrieval-compare:
	@echo "Running retrieval experiments..."
	$(PYTHON) -m app.retrieval compare \
		--cases $(RETRIEVAL_CASES) \
		--output-dir $(RETRIEVAL_OUTPUT) \
		--selection $(RETRIEVAL_SELECTION) \
		--chunk-unit $(RETRIEVAL_CHUNK_UNIT) \
		--chunk-overlap $(RETRIEVAL_CHUNK_OVERLAP) \
		--chunk-size $(RETRIEVAL_CHUNK_SIZE) \
		--top-k $(RETRIEVAL_TOP_K) \
		--build-indexes
	@echo "Retrieval experiments completed."

# Ensure Ollama is running and the embedding model is available
retrieval-ollama:
	@echo "Starting Ollama container..."
	docker compose $(COMPOSE_FLAGS) up -d ollama
	@echo "Waiting for embedding model: $(RETRIEVAL_EMBEDDING_MODEL)"
	@for i in $$(seq 1 180); do \
		if docker compose $(COMPOSE_FLAGS) exec -T ollama \
			ollama show "$(RETRIEVAL_EMBEDDING_MODEL)" >/dev/null 2>&1; then \
			echo "Embedding model is ready."; \
			exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "ERROR: Ollama model was not available after 6 minutes."; \
	exit 1