"""Routes for the nocapchicken Flask app."""

from __future__ import annotations

import dataclasses
import logging

from flask import Blueprint, jsonify, render_template, request

from .inference import predict

logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    """Serve the main search page."""
    return render_template("index.html")


@bp.post("/api/predict")
def api_predict():
    """
    Run model inference for a restaurant lookup.

    Request JSON: {"name": str, "city": str}
    Response JSON: PredictionResult as dict
    """
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


@bp.get("/health")
def health():
    """Liveness probe for deployment."""
    return jsonify({"status": "ok"})
