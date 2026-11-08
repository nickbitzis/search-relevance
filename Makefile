PORT ?= 8846

.PHONY: install test lint format run eval eval-gate debug screenshot docker-build clean

install:
	uv sync --python 3.12

test:
	uv run --python 3.12 --group dev pytest -q

lint:
	uvx ruff check .

format:
	uvx ruff format .
	uvx ruff check . --fix

run:  ## API and UI on http://127.0.0.1:$(PORT); startup builds the catalog, index, and models
	PORT=$(PORT) uv run searchrel serve

eval:  ## full evaluation, writes reports/evaluation.json and prints the tables
	uv run searchrel eval

eval-gate:  ## same, but exit 1 if the full pipeline's NDCG@10 is not above BM25 only
	uv run searchrel eval --gate

debug:  ## make debug Q="mens blue waterproof jacket under 200"
	uv run searchrel debug "$(Q)"

screenshot:  ## needs the server running on $(PORT) and Google Chrome installed (macOS path)
	"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
		--virtual-time-budget=20000 --window-size=1440,1300 --screenshot=docs/screenshot.png http://127.0.0.1:$(PORT)/
	"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
		--virtual-time-budget=20000 --window-size=1440,1500 --screenshot=docs/evaluation.png "http://127.0.0.1:$(PORT)/?view=eval"

docker-build:
	docker build -t search-relevance .

clean:
	rm -rf .pytest_cache .ruff_cache
