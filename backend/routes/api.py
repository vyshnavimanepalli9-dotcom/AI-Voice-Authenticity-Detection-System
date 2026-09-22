import os

from flask import Blueprint, request, jsonify, current_app

from backend.utils.security import (
    is_allowed_file,
    generate_safe_filename,
    validate_file_size,
    cleanup_old_uploads
)

from backend.services.audio_service import process_and_analyze_audio
from ml.predict import load_trained_model
from backend.database import save_analysis, get_analysis_history


api_bp = Blueprint('api', __name__)


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "service": "AI Voice Authenticity Detection Agent API",
        "version": "1.0.0"
    }), 200


@api_bp.route('/model-info', methods=['GET'])
def model_info():
    """Returns details about loaded ML model or demo state."""

    model, scaler, is_trained = load_trained_model()

    if is_trained:
        return jsonify({
            "status": "active",
            "model_type": type(model).__name__,
            "mode": "TRAINED MODEL MODE",
            "calibrated": True,
            "description": (
                "Scikit-Learn Random Forest Classifier trained on "
                "acoustic speech feature representations."
            )
        }), 200

    else:
        return jsonify({
            "status": "demo",
            "model_type": "Acoustic Signal Heuristic Evaluator",
            "mode": "DEMO MODE",
            "calibrated": False,
            "description": (
                "Operating in Demo Mode using pitch contour, spectral "
                "roll-off, and cepstral signal heuristics. Run "
                "'python ml/train.py' to generate a trained model binary."
            )
        }), 200


@api_bp.route('/upload', methods=['POST'])
def upload_audio():
    """
    Accepts audio file upload, validates format & size,
    and stores temporary file.
    """

    if 'audio' not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file provided in request."
        }), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400

    if not is_allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": (
                "Unsupported file format. Supported formats: "
                ".wav, .mp3, .ogg, .flac, .m4a, .webm"
            )
        }), 400

    valid_size, size_error = validate_file_size(file)

    if not valid_size:
        return jsonify({
            "success": False,
            "error": size_error
        }), 400

    upload_folder = current_app.config['UPLOAD_FOLDER']

    os.makedirs(upload_folder, exist_ok=True)
    cleanup_old_uploads(upload_folder)

    safe_filename = generate_safe_filename(file.filename)
    file_path = os.path.join(upload_folder, safe_filename)

    file.save(file_path)

    return jsonify({
        "success": True,
        "message": "File uploaded successfully.",
        "file_id": safe_filename,
        "original_filename": file.filename
    }), 200


@api_bp.route('/analyze', methods=['POST'])
@api_bp.route('/predict', methods=['POST'])
def analyze_audio():
    """
    Analyzes an audio file and returns authenticity results.
    """

    upload_folder = current_app.config['UPLOAD_FOLDER']

    file_path = None
    should_delete = True

    # ---------------------------------------------------------
    # CASE A: Audio file sent directly in form-data
    # ---------------------------------------------------------

    if 'audio' in request.files:

        file = request.files['audio']

        if not is_allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Unsupported file format."
            }), 400

        valid_size, size_error = validate_file_size(file)

        if not valid_size:
            return jsonify({
                "success": False,
                "error": size_error
            }), 400

        safe_filename = generate_safe_filename(file.filename)

        file_path = os.path.join(
            upload_folder,
            safe_filename
        )

        file.save(file_path)

    # ---------------------------------------------------------
    # CASE B: Previously uploaded file
    # ---------------------------------------------------------

    elif request.is_json and 'file_id' in request.json:

        file_id = request.json['file_id']

        file_path = os.path.join(
            upload_folder,
            file_id
        )

        if not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": "Uploaded file expired or not found."
            }), 404

        should_delete = False

    else:

        return jsonify({
            "success": False,
            "error": (
                "Invalid request. Provide 'audio' file "
                "or 'file_id'."
            )
        }), 400

    # ---------------------------------------------------------
    # Execute Audio Analysis & Prediction
    # ---------------------------------------------------------

    try:

        results = process_and_analyze_audio(file_path)

        # -----------------------------------------------------
        # Save analysis result to PostgreSQL
        # -----------------------------------------------------

        try:

            analysis_id = save_analysis(
                filename=os.path.basename(file_path),
                prediction=results.get("prediction"),
                is_human=results.get("is_human"),
                confidence=results.get("confidence"),
                demo_mode=results.get("demo_mode"),
                model_used=results.get("model_used"),
                status_note=results.get("status_note"),
                duration=results.get("duration"),
                sampling_rate=results.get("sampling_rate"),
                processing_time_ms=results.get(
                    "processing_time_ms"
                )
            )

            results["analysis_id"] = analysis_id

        except Exception as db_error:

            current_app.logger.warning(
                f"Database save failed: {db_error}"
            )

        return jsonify(results), 200

    except ValueError as ve:

        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400

    except Exception as e:

        current_app.logger.error(
            f"Analysis error: {e}"
        )

        return jsonify({
            "success": False,
            "error": (
                "An unexpected error occurred during "
                f"audio analysis: {str(e)}"
            )
        }), 500

    finally:

        # Clean up temporary uploaded file
        if (
            should_delete
            and file_path
            and os.path.exists(file_path)
        ):

            try:

                os.remove(file_path)

            except Exception as e:

                current_app.logger.warning(
                    f"Could not remove temp file {file_path}: {e}"
                )


# ============================================================
# ANALYSIS HISTORY
# ============================================================

@api_bp.route('/history', methods=['GET'])
def analysis_history():
    """Return recent voice analysis history."""

    try:

        limit = request.args.get(
            'limit',
            50,
            type=int
        )

        # Keep between 1 and 100 records
        limit = max(
            1,
            min(limit, 100)
        )

        history = get_analysis_history(limit)

        return jsonify({
            "success": True,
            "count": len(history),
            "history": history
        }), 200

    except Exception as e:

        current_app.logger.error(
            f"History retrieval error: {e}"
        )

        return jsonify({
            "success": False,
            "error": "Could not retrieve analysis history."
        }), 500