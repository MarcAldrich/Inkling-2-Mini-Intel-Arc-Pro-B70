# Makefile for Inkling-2-Mini (Unsloth UD-Q4_K_XL) on Intel Arc Pro B70
# Backend: TensorFold XPU research build (SergiioB/TensorFold @ b70-inkling2-mini-20261121, Level Zero)
# Serves OpenAI-compatible API on port 8080.

NAME           ?= inkling-b70-tf-xpu
PORT           ?= 8080
ALIAS          ?= inkling
IMAGE          ?= llama-b70-inkling:tf-xpu
MODEL_DIR      ?= $(abspath ..)
MODEL_FILE     ?= Inkling-2-Mini-UD-Q4_K_XL.gguf
MMPROJ_FILE    ?= mmproj-F16.gguf
CTX            ?= 262144
HEALTH_TIMEOUT ?= 300

.PHONY: build rebuild run stop logs status health smoke test test-http bench profile

build:
	@if podman image exists $(IMAGE) 2>/dev/null; then \
		echo "Image $(IMAGE) already exists — skipping build (use 'make rebuild' to force)."; \
	else \
		echo "Building $(IMAGE)..."; \
		podman build -f Containerfile -t $(IMAGE) .; \
	fi

rebuild:
	@echo "Rebuilding $(IMAGE) (no cache)..."
	podman build --no-cache -f Containerfile -t $(IMAGE) .

run:
	@test -f "$(MODEL_DIR)/$(MODEL_FILE)" || { echo "Missing model: $(MODEL_DIR)/$(MODEL_FILE)"; exit 1; }
	@PCI_ADDR=$$(lspci -D | grep -Ei 'vga|display' | grep -i battlemage | awk 'NR==1{print $$1}'); \
	if [ -z "$$PCI_ADDR" ]; then echo "No Battlemage GPU found via lspci"; exit 1; fi; \
	RENDER_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-render); \
	CARD_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-card); \
	echo "GPU: $$PCI_ADDR -> $$RENDER_NODE"; \
	MMPROJ_ARG=""; \
	if [ -f "$(MODEL_DIR)/$(MMPROJ_FILE)" ]; then MMPROJ_ARG="--mmproj /models/$(MMPROJ_FILE)"; fi; \
	podman rm -f $(NAME) >/dev/null 2>&1 || true; \
	podman run -d --name $(NAME) \
		-p $(PORT):8080 \
		--device $$RENDER_NODE --device $$CARD_NODE \
		--group-add keep-groups \
		--ulimit memlock=-1:-1 \
		--shm-size 2g \
		--stop-timeout 120 \
		-v $(MODEL_DIR):/models:ro \
		-e ZE_AFFINITY_MASK=0 \
		$(IMAGE) \
		--model /models/$(MODEL_FILE) \
		$$MMPROJ_ARG \
		--host 0.0.0.0 --port 8080 \
		--ctx $(CTX) \
		--name $(ALIAS)
	@echo "Container $(NAME) started. Waiting for /v1/models on port $(PORT)..."
	@for i in $$(seq 1 $$(($(HEALTH_TIMEOUT)/5))); do \
		if curl -fsS http://127.0.0.1:$(PORT)/v1/models >/dev/null 2>&1; then \
			echo ""; \
			echo "Model ready on port $(PORT)"; \
			exit 0; \
		fi; \
		if [ "$$(podman inspect -f '{{.State.Running}}' $(NAME) 2>/dev/null)" != "true" ]; then \
			echo ""; \
			echo "Container exited during startup:"; \
			podman logs --tail 60 $(NAME); \
			exit 1; \
		fi; \
		echo -n "."; \
		sleep 5; \
	done; \
	echo ""; \
	echo "Health timeout after $(HEALTH_TIMEOUT)s"; \
	podman logs --tail 60 $(NAME); \
	exit 1

# Direct runner decode benchmark (greedy, N_NEW tokens)
N_NEW ?= 256
bench:
	@test -f "$(MODEL_DIR)/$(MODEL_FILE)" || { echo "Missing model: $(MODEL_DIR)/$(MODEL_FILE)"; exit 1; }
	@PCI_ADDR=$$(lspci -D | grep -Ei 'vga|display' | grep -i battlemage | awk 'NR==1{print $$1}'); \
	RENDER_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-render); \
	CARD_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-card); \
	podman run --rm \
		--device $$RENDER_NODE --device $$CARD_NODE \
		--group-add keep-groups \
		--ulimit memlock=-1:-1 \
		-v $(MODEL_DIR):/models:ro \
		-e ZE_AFFINITY_MASK=0 \
		--entrypoint /bin/bash \
		$(IMAGE) -c 'printf "[1, 200001, 1000, 2000, 3000, 4000, 5000, 6000]\n" > /tmp/ids.json && /app/bin/xpu-inkling2 /models/$(MODEL_FILE) /tmp/ids.json $(N_NEW) $(CTX)'

# Direct runner with per-kernel GPU timestamp profiling (INK2_PROFILE=1)
profile:
	@test -f "$(MODEL_DIR)/$(MODEL_FILE)" || { echo "Missing model: $(MODEL_DIR)/$(MODEL_FILE)"; exit 1; }
	@PCI_ADDR=$$(lspci -D | grep -Ei 'vga|display' | grep -i battlemage | awk 'NR==1{print $$1}'); \
	RENDER_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-render); \
	CARD_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-card); \
	podman run --rm \
		--device $$RENDER_NODE --device $$CARD_NODE \
		--group-add keep-groups \
		--ulimit memlock=-1:-1 \
		-v $(MODEL_DIR):/models:ro \
		-e ZE_AFFINITY_MASK=0 \
		-e INK2_PROFILE=1 \
		--entrypoint /bin/bash \
		$(IMAGE) -c 'printf "[1, 200001, 1000, 2000, 3000, 4000, 5000, 6000]\n" > /tmp/ids.json && /app/bin/xpu-inkling2 /models/$(MODEL_FILE) /tmp/ids.json 64 $(CTX)'

stop:
	podman stop -t 15 $(NAME) >/dev/null 2>&1 || true
	podman rm -f $(NAME) >/dev/null 2>&1 || true
	@echo "Container $(NAME) stopped and removed"

logs:
	podman logs --tail 100 -f $(NAME)

status:
	@if podman container exists $(NAME) 2>/dev/null; then \
		podman inspect -f 'Running: {{.State.Running}}\nStarted: {{.State.StartedAt}}' $(NAME); \
	else \
		echo "Container $(NAME) not found"; \
	fi

health:
	curl -fsS http://127.0.0.1:$(PORT)/v1/models && echo " OK"

smoke:
	curl -fsS http://127.0.0.1:$(PORT)/v1/chat/completions \
		-H 'Content-Type: application/json' \
		-d '{"model":"'"$(ALIAS)"'","max_tokens":128,"temperature":1.0,"top_p":1.0,"messages":[{"role":"user","content":"What is 17*23? Reply with only the number."}]}'

# Direct GPU runner quality/parity/coherence regression suite (stop server first to avoid VRAM contention)
test:
	@test -f "$(MODEL_DIR)/$(MODEL_FILE)" || { echo "Missing model: $(MODEL_DIR)/$(MODEL_FILE)"; exit 1; }
	@PCI_ADDR=$$(lspci -D | grep -Ei 'vga|display' | grep -i battlemage | awk 'NR==1{print $$1}'); \
	RENDER_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-render); \
	CARD_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-card); \
	podman run --rm \
		--device $$RENDER_NODE --device $$CARD_NODE \
		--group-add keep-groups \
		--ulimit memlock=-1:-1 \
		-v $(MODEL_DIR):/models:ro \
		-e ZE_AFFINITY_MASK=0 \
		--entrypoint /opt/venv/bin/python3 \
		$(IMAGE) /app/test_quality.py --mode runner --runner /app/bin/xpu-inkling2 --model /models/$(MODEL_FILE) --meta /app/meta

# Live HTTP OpenAI + Hermes tool-calling + streaming + vision regression suite
test-http:
	podman run --rm --network host --entrypoint /opt/venv/bin/python3 \
		$(IMAGE) /app/test_quality.py --mode http --url http://127.0.0.1:$(PORT)
