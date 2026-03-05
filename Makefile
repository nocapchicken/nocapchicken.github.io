-include .env
export

.DEFAULT_GOAL := help

.PHONY: help install setup data-inspections data-google features train run run-prod lint test clean venv notebook

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
