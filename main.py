"""
main.py — Entry point for the nocapchicken web application.

Loads trained models and serves the inference API + frontend.
No training happens here.

Usage:
    python main.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from app import create_app


def main() -> None:
    """Launch the Flask application."""
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
