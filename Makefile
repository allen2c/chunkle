TEMPORAL_NAMESPACE ?= mybookshelf

# Development
format-all:
	isort .
	black .

update-all:
	poetry update
	poetry export \
		-f requirements.txt \
		--output requirements.txt \
		--without-hashes \
		--all-extras \
		--all-groups

# MCP
mcp-install:
	mcp install mcp_app.py -e .

mcp-dev:
	mcp dev mcp_app.py -e .

# Temporal
temporal-start:
	@echo "Starting temporal server with namespace $(TEMPORAL_NAMESPACE) ..."
	temporal server start-dev --namespace $(TEMPORAL_NAMESPACE)

temporal-stop:
	temporal server stop

temporal-worker:
	PYTHONPATH=. python scripts/temporal_worker.py

temporal-run-workflow:
	PYTHONPATH=. python scripts/temporal_run_workflow.py --book-id $(BOOK_ID) --chapter-id $(CHAPTER_ID)

test:
	pytest
