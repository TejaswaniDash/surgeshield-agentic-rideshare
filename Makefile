PY ?= python
export PYTHONPATH := src

.PHONY: install data data-tlc train demo eval test api worker temporal mcp up down clean all

install:            ## install dependencies
	pip install -r requirements.txt

data:               ## synthetic twin of HVFHV data (offline, ~20 s)
	$(PY) scripts/prepare_data.py --source synthetic

data-tlc:           ## REAL public NYC TLC HVFHV data (Uber = HV0003), ~450 MB / month
	$(PY) scripts/prepare_data.py --source tlc --months 2024-03

train:              ## LSTM forecaster, GBM wait model, anomaly detector, PPO policy, RAG index (~4 min CPU)
	$(PY) scripts/train_models.py

demo:               ## end-to-end incident, no servers needed
	$(PY) scripts/run_demo.py --scenario msg_letout

eval:               ## models + RAG + agent scenario suite -> artifacts/eval_report.md
	$(PY) scripts/evaluate.py

test:
	$(PY) -m pytest -q

api:                ## ops console at http://localhost:8000/ui (local runtime)
	uvicorn surgeshield.api.main:app --port 8000

temporal:           ## Temporal dev server (needs the `temporal` CLI)
	temporal server start-dev

worker:             ## Temporal worker
	$(PY) -m surgeshield.temporal_app.worker

mcp:                ## MCP server over stdio (point Claude Desktop / any MCP client at it)
	$(PY) -m surgeshield.mcp_server.server

up:                 ## full stack: Temporal + Postgres + API + worker + MCP
	docker compose up --build

down:
	docker compose down -v

clean:
	rm -rf artifacts/*.pt artifacts/*.pkl artifacts/*.zip artifacts/*.sqlite artifacts/*.jsonl data/processed/*

all: data train eval test
