# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Tennessee-specific routes using local data."""

from __future__ import annotations

import dataclasses
import logging

from flask import Blueprint, jsonify, render_template, request

from .inference_tn import predict, suggest_restaurants

logger = logging.getLogger(__name__)
bp_tn = Blueprint("tn", __name__)

# Default restaurant to show on page load
DEFAULT_RESTAURANT = "Hattie B's Hot Chicken"
DEFAULT_CITY = "Nashville"


@bp_tn.get("/")
def index():
    """Serve Tennessee version of the app with prefilled result."""
    try:
        prefill_result = predict(DEFAULT_RESTAURANT, DEFAULT_CITY)
        prefill_data = dataclasses.asdict(prefill_result)
    except Exception as exc:
        logger.warning("Prefill failed: %s", exc)
        prefill_data = None

    return render_template("index_tn.html", prefill_data=prefill_data)


@bp_tn.post("/api/predict")
def api_predict():
    """Expects JSON {"name": str, "city": str}."""
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    city = (body.get("city") or "").strip()

    if not name or not city:
        return jsonify({"error": "Both 'name' and 'city' are required."}), 400

    try:
        result = predict(name, city)
        result_dict = dataclasses.asdict(result)
        return jsonify(result_dict)
    except Exception as exc:
        logger.exception("Prediction failed for '%s' in '%s'", name, city)
        return jsonify({"error": str(exc)}), 500


@bp_tn.get("/api/suggest")
def api_suggest():
    """Expects ?name=str&city=str."""
    name = request.args.get("name", "").strip()
    city = request.args.get("city", "").strip()
    return jsonify(suggest_restaurants(name, city))


@bp_tn.get("/health")
def health():
    """Liveness probe."""
    return jsonify({"status": "ok"})


@bp_tn.get("/simple")
def simple():
    """Simple version without complex JS."""
    return render_template("simple_tn.html")
