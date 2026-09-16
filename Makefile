export HF_HOME=/home/$(USER)/goinfre/hf_cache
export UV_CACHE_DIR=/home/$(USER)/goinfre/uv_cache

.PHONY: install run clean

.venv/.installed:
	mkdir -p $(HF_HOME) $(UV_CACHE_DIR)
	uv sync
	touch .venv/.installed

install: .venv/.installed

run: install
	uv run python -m src

debug: install
	uv run python -m pdb -m src

clean:
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +

fclean: clean
	find . -type d -name ".venv" -exec rm -rf {} +
	rm -rf $(HF_HOME) $(UV_CACHE_DIR)

lint:
	uv run python -m flake8 src
	uv run python -m mypy src --warn-return-any \
		--warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run python -m flake8 src
	uv run python -m mypy src --strict
