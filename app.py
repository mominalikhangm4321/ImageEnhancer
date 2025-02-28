from flask import Flask, request, send_file
import os
import sys
import subprocess

# ==============================
# ✅ Manually Install Real-ESRGAN in Render
# ==============================

REALSRC_PATH = os.path.join(os.getcwd(), "Real-ESRGAN")
sys.path.append(REALSRC_PATH)

# If Real-ESRGAN is missing, clone it from GitHub and install dependencies
if not os.path.exists(REALSRC_PATH):
    print("Cloning Real-ESRGAN...")
    subprocess.run(["git", "clone", "https://github.com/xinntao/Real-ESRGAN.git"], check=True)

# Ensure dependencies are installed
print("Installing Real-ESRGAN dependencies...")
subprocess.run(["pip", "install", "-r", "Real-ESRGAN/requirements.txt"], check=True)
subprocess.run(["pip", "install", "basicsr"], check=True)
subprocess.run(["pip", "install", "facexlib"], check=True)
subprocess.run(["pip", "install", "gfpgan"], check=True)
subprocess.run(["pip", "install", "lmdb"], check=True)
subprocess.run(["pip", "install", "pyyaml"], check=True)
subprocess.run(["pip", "install", "yacs"], check=True)
subprocess.run(["pip", "install", "tqdm"], check=True)
subprocess.run(["pip", "install", "ffmpeg-python"], check=True)
subprocess.run(["pip", "install", "--no-cache-dir", "torchvision==0.15.2"], check=True)
subprocess.run(["pip", "install", "--no-cache-dir", "torch==2.0.1"], check=True)


from realesrgan.utils import RealESRGANer  # Import after ensuring installation

# ==============================
# ✅ Flask App Setup
# ==============================

from enhancer import enhance_image
from flask_cors import CORS  # Import Flask-CORS

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
