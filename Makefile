-include .env
export

.DEFAULT_GOAL := help

.PHONY: help install setup data-inspections data-google features train run run-prod lint test clean venv notebook colab-pat

help:
	@echo ""
	@echo "\033[2m# Setup\033[0m"
	@echo "  \033[36mvenv\033[0m       Create .venv and install all dependencies"
	@echo "  \033[36mnotebook\033[0m   Launch Jupyter in notebooks/"
	@echo "  \033[36minstall\033[0m    Install Python dependencies (no venv)"
	@echo "  \033[36msetup\033[0m      Collect inspections + reviews (reads .env)"
	@echo ""
	@echo "\033[2m# Data\033[0m"
	@echo "  \033[36mfeatures\033[0m   Build feature matrix from raw data"
	@echo "  \033[36mtrain\033[0m      Train baseline + RF (fast, skips DistilBERT)"
	@echo ""
	@echo "\033[2m# App\033[0m"
	@echo "  \033[36mrun\033[0m        Start Flask dev server on :5000"
	@echo "  \033[36mrun-prod\033[0m   Start with gunicorn (production)"
	@echo ""
	@echo "\033[2m# Dev\033[0m"
	@echo "  \033[36mlint\033[0m       Run ruff linter"
	@echo "  \033[36mtest\033[0m       Run pytest"
	@echo "  \033[36mclean\033[0m      Remove __pycache__ and .pyc files"
	@echo "  \033[36mcolab-pat\033[0m  Create a GitHub PAT for Colab notebook publishing"
	@echo ""
	@echo "\033[2m# data-inspections / data-google  — individual collection steps\033[0m"
	@echo ""

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt jupyter ipykernel
	.venv/bin/python -m ipykernel install --user --name nocapchicken --display-name "nocapchicken"
	@echo "\033[32m  venv ready — run: make notebook\033[0m"

notebook:
	.venv/bin/jupyter notebook notebooks/

install:
	pip3 install -r requirements.txt

setup:
	python3 setup.py

data-inspections:
	python3 scripts/make_dataset.py --inspections-only

data-google:
	python3 scripts/make_dataset.py --google-key $${GOOGLE_PLACES_API_KEY:?set GOOGLE_PLACES_API_KEY in .env}


features:
	python3 scripts/build_features.py

train:
	python3 scripts/model.py --skip-bert

run:
	@printf "\n\033[1;36m  nocapchicken: http://localhost:5000\033[0m\n\n"
	FLASK_ENV=development python3 main.py

run-prod:
	@printf "\n\033[1;36m  nocapchicken: http://0.0.0.0:5000\033[0m\n\n"
	gunicorn --bind 0.0.0.0:5000 --workers 2 "app:create_app()"

lint:
	ruff check .

test:
	pytest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete

# Fine-grained PAT for Colab → GitHub push.
# GitHub ignores the target_name param, so the user must change Resource Owner manually.
GITHUB_COLAB_PAT_URL := https://github.com/settings/personal-access-tokens/new?name=nocapchicken%20Colab&description=Colab%20token%20for%20nocapchicken.github.io&expires_in=30&contents=write

colab-pat:
	@echo ""
	@echo "  \033[1;33m⚠  IMPORTANT: Change Resource Owner to nocapchicken\033[0m"
	@echo ""
	@echo "  GitHub defaults to your personal account. The token will NOT work"
	@echo "  unless you change the Resource Owner dropdown at the top of the form"
	@echo "  from your username to \033[1mnocapchicken\033[0m before creating."
	@echo ""
	@echo "  Save the token in Colab secrets as \033[1mGITHUB_TOKEN_NOCAPCHICKEN\033[0m."
	@echo ""
	@url='$(GITHUB_COLAB_PAT_URL)'; \
	if command -v open >/dev/null 2>&1; then \
		open "$$url"; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open "$$url"; \
	else \
		printf '%s\n' "$$url"; \
	fi
