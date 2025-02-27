from flask import Flask, request, send_file
import os
import sys
import subprocess
from enhancer import enhance_image
from flask_cors import CORS  # Import Flask-CORS

# ==============================
# ✅ Manually Add Real-ESRGAN
# ==============================

REALSRC_PATH = os.path.join(os.getcwd(), "Real-ESRGAN")
sys.path.append(REALSRC_PATH)

# If Real-ESRGAN is missing, clone it from GitHub and install dependencies
if not os.path.exists(REALSRC_PATH):
    print("Cloning Real-ESRGAN...")
    subprocess.run(["git", "clone", "https://github.com/xinntao/Real-ESRGAN.git"], check=True)
    print("Installing Real-ESRGAN dependencies...")
    subprocess.run(["pip", "install", "-r", "Real-ESRGAN/requirements.txt"], check=True)

from realesrgan.utils import RealESRGANer  # Import after ensuring it's installed

# ==============================
# ✅ Flask App Setup
# ==============================

app = Flask(__name__)
CORS(app)  # Enable CORS

UPLOAD_FOLDER = "static/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["image"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    enhanced_path = enhance_image(file_path)
    
    return send_file(enhanced_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
