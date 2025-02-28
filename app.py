from flask import Flask, request, send_file
import os
import sys
import subprocess
import urllib.request

# ==============================
# ✅ Ensure Real-ESRGAN is Installed & Model is Downloaded
# ==============================

REALSRC_PATH = os.path.abspath(os.path.join(os.getcwd(), "Real-ESRGAN"))  # Ensure absolute path
WEIGHTS_PATH = os.path.join(REALSRC_PATH, "weights")
MODEL_FILE = os.path.join(WEIGHTS_PATH, "RealESRGAN_x4plus.pth")
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

sys.path.append(REALSRC_PATH)

# ✅ Fix: Ensure 'Real-ESRGAN' directory exists before cloning
if not os.path.exists(REALSRC_PATH):
    print("Cloning Real-ESRGAN...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/xinntao/Real-ESRGAN.git"], check=True)

# ✅ Fix: Ensure the weights directory exists
os.makedirs(WEIGHTS_PATH, exist_ok=True)

# ✅ Fix: Force download the model file if it's missing
if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) == 0:
    print(f"Downloading model weights to: {MODEL_FILE}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("✅ Model downloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to download model weights: {e}")
        sys.exit(1)  # Stop execution if download fails

# ✅ Fix: Ensure dependencies are installed properly
print("Installing Real-ESRGAN dependencies...")
subprocess.run(["pip", "install", "-r", os.path.join(REALSRC_PATH, "requirements.txt")], check=True)

# ✅ Fix: Ensure 'version.py' exists to prevent import errors
VERSION_FILE = os.path.join(REALSRC_PATH, "realesrgan/version.py")
if not os.path.exists(VERSION_FILE):
    print("Creating missing 'version.py' file for Real-ESRGAN...")
    with open(VERSION_FILE, "w") as f:
        f.write("__version__ = '0.3.0'\n")

# ✅ Fix: Install required dependencies with proper versions
print("Installing required dependencies...")
dependencies = [
    "basicsr",
    "facexlib",
    "gfpgan",
    "lmdb",
    "pyyaml",
    "yacs",
    "tqdm",
    "ffmpeg-python",
    "torchvision==0.15.2",
    "torch==2.0.1",
    "numpy<2"
]

for dep in dependencies:
    subprocess.run(["pip", "install", "--no-cache-dir", dep], check=True)

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
