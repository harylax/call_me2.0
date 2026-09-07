# export HF_HOME=/goinfre/$(USER)/hf_cache
# export UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/.venv
# export UV_CACHE_DIR=/goinfre/$(USER)/uv_cache
# export HF_HOME=/sgoinfre/$(USER)/hf_cache
# export UV_PROJECT_ENVIRONMENT=/sgoinfre/$(USER)/.venv
# export UV_CACHE_DIR=/sgoinfre/$(USER)/uv_cache
export HF_HOME=../hf_cache
export UV_CACHE_DIR=../uv_cache

.PHONY: install run clean


$(UV_PROJECT_ENVIRONMENT)/.installed:
	mkdir -p $(HF_HOME) $(UV_CACHE_DIR)
	uv sync
	touch $(UV_PROJECT_ENVIRONMENT)/.installed

install: $(UV_PROJECT_ENVIRONMENT)/.installed

run: install
	uv run python3 -m src

clean:
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +