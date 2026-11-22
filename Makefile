# Makefile for Inkling-2-Mini (Unsloth UD-Q4_K_XL) on Intel Arc Pro B70
# Backend: vengeful180-dot/llama.cpp Fused Vulkan (Mesa ANV)
# Serves OpenAI-compatible API on port 8080.

NAME           ?= inkling-b70-vk-fused
PORT           ?= 8080
ALIAS          ?= inkling
IMAGE          ?= llama-b70-inkling:vk-fused
MODEL_DIR      ?= $(abspath .)
MODEL_FILE     ?= Inkling-2-Mini-UD-Q4_K_XL.gguf
MMPROJ_FILE    ?=
CTX            ?= 65536
HEALTH_TIMEOUT ?= 300

MMPROJ_ARGS := $(if $(MMPROJ_FILE),--mmproj /models/$(MMPROJ_FILE) --no-mmproj-offload,)

.PHONY: build rebuild run stop logs status health smoke test-ops bench

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
	podman rm -f $(NAME) >/dev/null 2>&1 || true; \
	podman run -d --name $(NAME) \
		-p $(PORT):8080 \
		--device $$RENDER_NODE --device $$CARD_NODE \
		--group-add keep-groups \
		--ulimit memlock=-1:-1 \
		--shm-size 2g \
		--stop-timeout 120 \
		-v $(MODEL_DIR):/models:ro \
		-e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json \
		$(IMAGE) \
		--model /models/$(MODEL_FILE) \
		$(MMPROJ_ARGS) \
		--alias $(ALIAS) \
		--host 0.0.0.0 --port 8080 \
		--n-gpu-layers 999 \
		--ctx-size $(CTX) \
		--batch-size 1024 --ubatch-size 512 \
		--parallel 1 \
		--flash-attn on \
		--cache-type-k f16 --cache-type-v f16 \
		--jinja \
		--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
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

# Verify all fused Vulkan ops against CPU reference on B70
test-ops:
	@PCI_ADDR=$$(lspci -D | grep -Ei 'vga|display' | grep -i battlemage | awk 'NR==1{print $$1}'); \
	RENDER_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-render); \
	CARD_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-card); \
	podman run --rm \
		--device $$RENDER_NODE --device $$CARD_NODE \
		--group-add keep-groups \
		-e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json \
		--entrypoint /app/llama.cpp/build/bin/test-backend-ops \
		$(IMAGE) test -b Vulkan0

# Quick llama-bench (pp512 / tg64 + 32k depth) — stop server first to free VRAM
bench:
	@test -f "$(MODEL_DIR)/$(MODEL_FILE)" || { echo "Missing model: $(MODEL_DIR)/$(MODEL_FILE)"; exit 1; }
	@PCI_ADDR=$$(lspci -D | grep -Ei 'vga|display' | grep -i battlemage | awk 'NR==1{print $$1}'); \
	RENDER_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-render); \
	CARD_NODE=$$(readlink -f /dev/dri/by-path/pci-$$PCI_ADDR-card); \
	podman run --rm \
		--device $$RENDER_NODE --device $$CARD_NODE \
		--group-add keep-groups \
		-v $(MODEL_DIR):/models:ro \
		-e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json \
		--entrypoint /app/llama.cpp/build/bin/llama-bench \
		$(IMAGE) -m /models/$(MODEL_FILE) -ngl 99 -fa 1 -ctk f16 -ctv f16 -p 512 -n 64 -d 0,32768

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
		-d '{"model":"'"$(ALIAS)"'","max_tokens":32,"temperature":0,"messages":[{"role":"user","content":"What is 17*23? Reply with only the number."}]}'
