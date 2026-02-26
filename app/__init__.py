"""Flask application factory for nocapchicken."""

from flask import Flask

from .routes import bp


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.register_blueprint(bp)
    return app
