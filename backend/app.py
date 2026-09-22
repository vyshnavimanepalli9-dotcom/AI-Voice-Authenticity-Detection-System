import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from backend.routes.api import api_bp
from backend.database import init_database

# Set absolute path to frontend directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app) # Enable Cross-Origin Resource Sharing

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max payload limit
init_database() # Initialize the database and create tables if they don't exist

# Register API routes blueprint
app.register_blueprint(api_bp, url_prefix='/api')

@app.route('/')
def serve_index():
    """Serves main UI landing page."""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serves frontend static assets (CSS, JS, icons)."""
    return send_from_directory(FRONTEND_DIR, filename)

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("\n=======================================================")
    print("  AI VOICE AUTHENTICITY DETECTION AGENT - SERVER RUNNING ")
    print("  Access Web Application: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(host='127.0.0.1', port=5000, debug=True)
