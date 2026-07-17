# Development
fmt:
	isort chunkle tests
	black chunkle tests

update:
	poetry update

test:
	pytest
