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
    return render_template("index.html")


@bp.post("/api/predict")
def api_predict():
    """Run model inference. Expects JSON {"name": str, "city": str}."""
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    city = (body.get("city") or "").strip()

    if not name or not city:
        return jsonify({"error": "Both 'name' and 'city' are required."}), 400

    try:
        result = predict(name, city)
        return jsonify(dataclasses.asdict(result))
    except Exception as exc:
        logger.exception("Prediction failed for '%s' in '%s'", name, city)
        return jsonify({"error": str(exc)}), 500


@bp.get("/api/suggest")
def api_suggest():
    """Return restaurant name suggestions. Expects ?name=str&city=str."""
    name = request.args.get("name", "").strip()
    city = request.args.get("city", "").strip()
    return jsonify(suggest_restaurants(name, city))


@bp.get("/health")
def health():
    """Liveness probe for deployment."""
    return jsonify({"status": "ok"})
