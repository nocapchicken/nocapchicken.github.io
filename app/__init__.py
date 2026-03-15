# AI-assisted (Claude Code, claude.ai) — https://claude.ai
# External libraries: Flask (BSD-3), Flask-CORS (MIT)
from flask import Flask
from flask_cors import CORS

from .routes import bp


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    CORS(app, resources={r"/api/*": {"origins": ["https://nocapchicken.github.io", "https://nocapchicken-github-io.onrender.com"]}})
    app.register_blueprint(bp)
    return app
