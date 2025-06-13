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

test:
	pytest
