from flask import Flask, request, send_file
import os
from enhancer import enhance_image
from flask_cors import CORS
import os
import sys

# Manually add Real-ESRGAN path
sys.path.append(os.path.abspath("Real-ESRGAN"))

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["image"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Enhance the image
    enhanced_path = enhance_image(file_path)
    
    return send_file(enhanced_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
