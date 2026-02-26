.DEFAULT_GOAL := help

.PHONY: help install setup data features train run run-prod lint test clean

help:
	@echo ""
	@echo "Setup"
	@echo "  \033[36minstall\033[0m      Install Python dependencies"
	@echo "  \033[36msetup\033[0m        Collect NC inspections + Yelp + Google data"
	@echo ""
	@echo "Data"
	@echo "  \033[36mdata\033[0m         Re-run data collection only"
	@echo "  \033[36mfeatures\033[0m     Build feature matrix from raw data"
	@echo ""
	@echo "Train"
	@echo "  \033[36mtrain\033[0m        Train all three models (baseline, RF, DistilBERT)"
	@echo ""
	@echo "App"
	@echo "  \033[36mrun\033[0m          Start Flask dev server on :5000"
	@echo "  \033[36mrun-prod\033[0m     Start Flask with gunicorn (production)"
	@echo ""
	@echo "Dev"
	@echo "  \033[36mlint\033[0m         Run ruff linter"
	@echo "  \033[36mtest\033[0m         Run pytest"
	@echo "  \033[36mclean\033[0m        Remove __pycache__ and .pyc files"
	@echo ""

install:
	pip3 install -r requirements.txt

setup:
	python3 setup.py

data:
	python3 scripts/make_dataset.py

features:
	python3 scripts/build_features.py

train:
	python3 scripts/model.py

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
