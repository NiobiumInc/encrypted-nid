# ==============================================================================
# FHE Network Monitor - Makefile (OSS / niobium-client transport flow)
# ==============================================================================
# Thin convenience wrapper. The PRIMARY entry point is the harness:
#
#     python3 harness/run_submission.py -h
#
# which builds the SDK stages on demand (scripts/build_task.sh) and drives
# record/replay over the niobium-client FHETCH transport. This Makefile only
# adds shortcuts for building, reporting, and cleaning. See NIOBIUM_INTEGRATION.md.

# ==============================================================================
# Directory Configuration
# ==============================================================================
ROOT_DIR := $(shell pwd)
SCRIPTS_DIR := $(ROOT_DIR)/scripts
BUILD_DIR := $(ROOT_DIR)/build
ASSETS_DIR := $(ROOT_DIR)/assets
WORKLOAD_INPUTS_DIR := $(ROOT_DIR)/Mirai_Workload_Inputs

# ==============================================================================
# PHONY Target Declarations
# ==============================================================================
.PHONY: help build report info
.PHONY: clean clean-build clean-runs clean-keys clean-cache clean-workload-inputs clean-all distclean

.DEFAULT_GOAL := help

# ==============================================================================
# Help Target (Self-Documenting)
# ==============================================================================
help: ## Display this help message
	@echo ""
	@echo "FHE Network Monitor - Available Targets"
	@echo "========================================"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n\n"} \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 } \
		END { printf "\n" }' $(MAKEFILE_LIST)
	@echo "Primary entry point (builds on demand, no make target needed):"
	@echo "  python3 harness/run_submission.py -h"
	@echo ""

##@ Build Targets

build: ## Build the SDK stages (delegates to scripts/build_task.sh; the harness also builds on demand)
	@bash $(SCRIPTS_DIR)/build_task.sh

##@ Reporting

report: ## Charts from a run artifact. Usage: make report RUN=runs/<id> [PROFILE=<p>] [CPU=<cpu_scores.csv>]
	@test -n "$(RUN)" || { echo "usage: make report RUN=runs/<id> [PROFILE=<p>] [CPU=<cpu_scores.csv>]"; exit 1; }
	@P="$(PROFILE)"; [ -n "$$P" ] || P=$$(python3 -c "import json;print(json.load(open('$(RUN)/run.json'))['profile'])"); \
	 ACC=$$(python3 -c "import json;print(json.load(open('profiles.json'))['profiles'].get('$$P',{}).get('accuracy_profile',False))"); \
	 if [ "$$ACC" != "True" ]; then \
	   echo "[report] profile '$$P' is a functionality check — the run already reported VALIDATION PASSED (decrypt == plaintext)."; \
	   echo "[report] The detection report (plaintext-vs-FPGA + precision/recall) is for the real workload:"; \
	   echo "[report]   make report RUN=<a '--profile mirai' run>"; \
	   exit 0; \
	 fi; \
	 python3 $(SCRIPTS_DIR)/plaintext_reference.py "$(RUN)" --profile "$$P" && \
	 echo "Exact-activation plaintext reference -> $(RUN)/reference_batch*.csv"; \
	 python3 $(SCRIPTS_DIR)/plot_fpga_results.py "$(RUN)" --profile "$$P" --save-figs --hide-figs && \
	 echo "FPGA anomaly charts -> $(RUN)/graphs/"; \
	 CPU_ARG=""; [ -n "$(CPU)" ] && CPU_ARG="--cpu $(CPU)"; \
	 python3 $(SCRIPTS_DIR)/plot_comparison.py "$(RUN)" "$(RUN)/plots" --profile "$$P" $$CPU_ARG && \
	 echo "Plaintext-vs-FPGA comparison -> $(RUN)/plots/ (add CPU=<cpu_scores.csv> for a CPU-FHE overlay)"; \
	 python3 $(SCRIPTS_DIR)/demo_report.py "$(RUN)" --profile "$$P" || true

##@ Cleaning Targets

clean: clean-build ## Remove build artifacts only
	@echo "✅ Build artifacts cleaned."

clean-build: ## Remove the SDK build directory
	@echo "Removing build artifacts..."
	@rm -rf $(BUILD_DIR) || true

clean-runs: ## Remove runtime data (workload directories, runs/, .concurrent/)
	@echo "Removing runtime data..."
	@rm -rf $(ROOT_DIR)/NID_Mirai_*_workload* || true
	@rm -rf $(ROOT_DIR)/Encrypted_Mirai_* || true
	@rm -rf $(ROOT_DIR)/.concurrent || true
	@rm -rf $(ROOT_DIR)/runs || true
	@echo "✅ Runtime data cleaned."

clean-keys: ## Remove crypto keys + intermediate ciphertexts (preserve models/norms)
	@echo "Removing crypto keys from Mirai_Workload_Inputs/..."
	@rm -f $(WORKLOAD_INPUTS_DIR)/cryptocontext.bin
	@rm -f $(WORKLOAD_INPUTS_DIR)/secret_key.bin
	@rm -f $(WORKLOAD_INPUTS_DIR)/public_key.bin
	@rm -f $(WORKLOAD_INPUTS_DIR)/relinearization_key.bin
	@rm -f $(WORKLOAD_INPUTS_DIR)/score_ciphertext_*.bin
	@rm -f $(WORKLOAD_INPUTS_DIR)/feature_ciphertext_*.bin
	@echo "✅ Keys cleaned (models and norms preserved)"

clean-cache: ## Remove global key cache directories
	@echo "Removing cache directories..."
	@rm -rf $(ROOT_DIR)/global_key_cache
	@rm -rf $(ROOT_DIR)/global_key_cache_*
	@echo "✅ Cache cleaned"

clean-workload-inputs: ## Remove entire Mirai_Workload_Inputs directory
	@echo "⚠️  WARNING: This will remove ALL workload inputs!"
	@echo "Press Ctrl+C within 3 seconds to cancel..."
	@sleep 3
	@rm -rf $(WORKLOAD_INPUTS_DIR)
	@echo "✅ Workload inputs cleaned."

clean-all: clean-build clean-runs clean-keys clean-cache ## Deep clean (everything except source assets)
	@if [ -d "$(WORKLOAD_INPUTS_DIR)" ] && [ -z "$$(ls -A $(WORKLOAD_INPUTS_DIR) 2>/dev/null)" ]; then \
		rmdir "$(WORKLOAD_INPUTS_DIR)" 2>/dev/null && echo "ℹ️  Removed empty $(WORKLOAD_INPUTS_DIR)/" || true; \
	fi
	@echo "✅ Deep clean complete."

distclean: clean-all clean-workload-inputs ## Nuclear clean (also removes Mirai_Workload_Inputs/)
	@echo "✅ Complete clean finished."

##@ Information Targets

info: ## Display configuration + status
	@echo ""
	@echo "FHE Network Monitor - Status"
	@echo "============================"
	@echo "Root:            $(ROOT_DIR)"
	@echo "SDK build dir:   $(BUILD_DIR)"
	@echo ""
	@if [ -x "$(BUILD_DIR)/server_standalone_sdk" ]; then \
		echo "  SDK stages:    ✅ Built"; \
	else \
		echo "  SDK stages:    ❌ Not built (run 'make build' or the harness)"; \
	fi
	@if [ -f "$(ASSETS_DIR)/models/Mirai_model_FULL.bin" ]; then \
		echo "  FULL assets:   ✅ Present"; \
	else \
		echo "  FULL assets:   ❌ Missing"; \
	fi
	@echo ""
