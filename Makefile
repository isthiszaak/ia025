.PHONY: up down clean-build logs clean-all pull-models fix-port

# Detect if an NVIDIA GPU is available on the host machine
HAS_GPU := $(shell nvidia-smi > /dev/null 2>&1 && echo yes || echo no)

# Defines docker compose flags based on GPU availability
COMPOSE_FLAGS := -f docker-compose.cpu.yaml
ifeq ($(HAS_GPU),yes)
    COMPOSE_FLAGS += -f docker-compose.gpu.yaml
endif

# Brings up the containers in the background
up:
	@if [ "$(HAS_GPU)" = "yes" ]; then \
		echo "NVIDIA GPU detected! Starting containers with GPU acceleration..."; \
	else \
		echo "No NVIDIA GPU detected. Starting containers in CPU mode..."; \
	fi
	docker compose $(COMPOSE_FLAGS) up -d
	@echo "All containers have been started. Check if OLLAMA is pulling the model in the Ollama container. This may take a few minutes."

# Brings down the containers and explicitly removes orphans
down:
	docker compose $(COMPOSE_FLAGS) down
	@echo "All containers have been stopped."

# Brings down, removes orphans, forces a rebuild of the images, and brings up
rebuild:
	@echo "Rebuilding all containers..."
	docker compose $(COMPOSE_FLAGS) up -d --build
	@echo "All containers have been rebuilt and started."
	@echo "If the Ollama container has been rebuilt, wait for Ollama to finish pulling the model before using the frontend. This may take a few minutes."
	
# Tails the logs for all services
logs:
	docker compose $(COMPOSE_FLAGS) logs -f

# Deep clean: stops containers, removes orphans, and deletes volumes (resets database/models)
clean-all:
	docker compose $(COMPOSE_FLAGS) down --remove-orphans --volumes
	@echo "Deep clean complete. All containers, networks, and volumes have been removed."
