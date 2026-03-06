# AI-assisted (Claude Code, claude.ai) — https://claude.ai
from __future__ import annotations

import dataclasses
import logging

from flask import Blueprint, jsonify, render_template, request

from .inference import predict, suggest_restaurants

logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    """Render the main search page."""
    return render_template("index.html")


@bp.post("/api/predict")
def api_predict():
    """Expects JSON {"name": str}."""
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()

    if not name:
        return jsonify({"error": "'name' is required."}), 400

    try:
        result = predict(name)
        return jsonify(dataclasses.asdict(result))
    except Exception as exc:
        logger.exception("Prediction failed for '%s'", name)
        return jsonify({"error": str(exc)}), 500


@bp.get("/api/suggest")
def api_suggest():
    """Expects ?name=str."""
    name = request.args.get("name", "").strip()
    return jsonify(suggest_restaurants(name))


@bp.get("/health")
def health():
    """Liveness probe for deployment."""
    return jsonify({"status": "ok"})
