# Define variables for commands to keep the Makefile clean
PIP_INSTALL = pip install
CONDA_INSTALL = conda install -y --copy
GIT_CLONE = git clone
GIT_SUBMODULE = git submodule
CD = cd
ECHO = echo

# The default target
all: install_deps init_and_update_submodules clone_and_install_fastchat check_gpu

# Install Python dependencies
install_deps:
	$(PIP_INSTALL) pandas==2.2.1 sqlparse==0.4.4 levenshtein==0.25.0 rouge-score==0.1.2 datasets==2.18.0 shortuuid==1.0.13 alpaca-eval==0.6 openai==1.14.3

# Initialize and update submodules
init_and_update_submodules:
	$(GIT_SUBMODULE) update --init --recursive

# Clone FastChat and install it
clone_and_install_fastchat:
	$(CD) FastChat && \
	$(PIP_INSTALL) -e ".[model_worker,llm_judge]"

# Check if GPU exists and install CUDA or llama-cpp-python accordingly
check_gpu:
	@if [ -x "$$(command -v nvidia-smi)" ]; then \
		$(ECHO) "GPU detected, installing CUDA-specific packages."; \
		$(CONDA_INSTALL) -c "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-nvcc && \
		CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 $(PIP_INSTALL) --upgrade --force-reinstall llama-cpp-python --no-cache-dir; \
	else \
		$(ECHO) "No GPU detected, installing CPU version of llama-cpp-python."; \
		$(PIP_INSTALL) llama-cpp-python; \
	fi

.PHONY: all install_deps init_and_update_submodules clone_and_install_fastchat check_gpu
