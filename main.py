# AI-assisted (Claude Code, claude.ai) — https://claude.ai

from app import create_app


def main() -> None:
    """Create the Flask app and start the development server."""
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
