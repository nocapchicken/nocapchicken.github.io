# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Run the Tennessee local version of nocapchicken (no API keys needed)."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from flask import Flask
from flask_cors import CORS

from app.routes_tn import bp_tn


def create_tn_app() -> Flask:
    """Create Flask app with Tennessee routes."""
    app = Flask(
        __name__,
        template_folder="app/templates",
        static_folder="app/static"
    )
    CORS(app)
    app.register_blueprint(bp_tn)
    return app


def main() -> None:
    """Start the Tennessee version dev server."""
    app = create_tn_app()
    print("\n  nocapchicken (Tennessee Edition): http://localhost:5001\n")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)


if __name__ == "__main__":
    main()
