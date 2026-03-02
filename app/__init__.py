# AI-assisted (Claude Code, claude.ai) — https://claude.ai
from flask import Flask
from flask_cors import CORS

from .routes import bp


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    CORS(app, resources={r"/api/*": {"origins": ["https://nocapchicken.github.io", "https://nocapchicken-github-io.onrender.com"]}})
    app.register_blueprint(bp)
    return app
