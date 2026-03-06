"""
HemoScan AI — Flask Backend (Groq Version)
============================================================
Model  : RandomForestClassifier (model5.pkl)
Features → ['Gender','Hemoglobin','MCH','MCHC','MCV']

AI Extraction → Groq Llama3
============================================================
"""

import os
import re
import json
import pickle
import warnings
import pytesseract
import pdfplumber
import pandas as pd

from PIL import Image
from groq import Groq
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load .env file
load_dotenv()

app = Flask(__name__, static_folder=".", static_url_path="")

# ==========================================================
# Load RandomForest Model
# ==========================================================

MODEL_PATH = "model5.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        rf_model = pickle.load(f)

    print("✓ Model loaded successfully")
    print("Features:", rf_model.feature_names_in_)

except Exception as e:

    rf_model = None
    print("Model loading failed:", e)

# ==========================================================
# Configure Groq
# ==========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✓ Groq configured")
else:
    groq_client = None
    print("⚠ GROQ_API_KEY not set")

# ==========================================================
# Helper Functions
# ==========================================================

def encode_gender(gender):
    gender = gender.lower()
    if gender == "female":
        return 1
    else:
        return 0


def anemia_subtype(mcv, mch):

    if mcv < 80 and mch < 27:
        return "microcytic hypochromic anemia"

    if mcv > 100:
        return "macrocytic anemia"

    return "normocytic anemia"


def build_features(g, hb, mch, mchc, mcv):

    return pd.DataFrame(
        [[g, hb, mch, mchc, mcv]],
        columns=["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
    )

# ==========================================================
# Text Extraction
# ==========================================================

def extract_text_from_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def extract_text_from_image(file):

    img = Image.open(file).convert("RGB")

    text = pytesseract.image_to_string(img)

    return text

# ==========================================================
# Groq CBC Extraction
# ==========================================================

def extract_cbc_with_groq(text):

    prompt = f"""
Extract these CBC values from the medical report.

Return ONLY JSON.

Fields:
Hemoglobin
MCV
MCH
MCHC

Format:
{{
"hemoglobin": number or null,
"mcv": number or null,
"mch": number or null,
"mchc": number or null
}}

Report:
{text}
"""

    response = groq_client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[{
            "role": "user",
            "content": prompt
        }],

        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    # Remove markdown fences
    raw = re.sub(r"```json", "", raw)
    raw = re.sub(r"```", "", raw)

    try:
        return json.loads(raw)

    except:
        return {
            "hemoglobin": None,
            "mcv": None,
            "mch": None,
            "mchc": None
        }

# ==========================================================
# Routes
# ==========================================================

@app.route("/")
def index():

    return send_from_directory(".", "index.html")

# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.route("/predict", methods=["POST"])

def predict():

    if rf_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()

    try:

        hb = float(data["hemoglobin"])
        mcv = float(data["mcv"])
        mch = float(data["mch"])
        mchc = float(data["mchc"])

        gender = encode_gender(data.get("gender", ""))

    except:

        return jsonify({"error": "Invalid inputs"}), 400

    features = build_features(gender, hb, mch, mchc, mcv)

    pred = rf_model.predict(features)[0]

    proba = rf_model.predict_proba(features)[0]

    is_anemia = pred == 1

    confidence = float(proba[1] if is_anemia else proba[0]) * 100

    if is_anemia:

        subtype = anemia_subtype(mcv, mch)

        message = f"Model predicts {subtype} with {confidence:.1f}% confidence."

    else:

        message = f"No anemia detected with {confidence:.1f}% confidence."

    return jsonify({

        "prediction": "Anemia Detected" if is_anemia else "No Anemia",

        "confidence": round(confidence, 2),

        "message": message,

        "probabilities": {
            "no_anemia": float(proba[0]),
            "anemia": float(proba[1])
        }

    })

# ==========================================================
# Extract CBC Values
# ==========================================================

@app.route("/extract", methods=["POST"])
def extract():

    if groq_client is None:
        return jsonify({"error": "Groq API not configured"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    filename = file.filename.lower()

    try:

        if filename.endswith(".pdf"):

            text = extract_text_from_pdf(file)

        else:

            text = extract_text_from_image(file)

        data = extract_cbc_with_groq(text)

        return jsonify({

            "success": True,

            "hemoglobin": data.get("hemoglobin"),
            "mcv": data.get("mcv"),
            "mch": data.get("mch"),
            "mchc": data.get("mchc")

        })

    except Exception as e:

        return jsonify({"error": str(e)}), 500

# ==========================================================
# Model Metadata
# ==========================================================

@app.route("/model-info")
def model_info():

    if rf_model is None:
        return jsonify({"error": "Model not loaded"})

    return jsonify({

        "model": type(rf_model).__name__,

        "features": list(rf_model.feature_names_in_),

        "classes": rf_model.classes_.tolist(),

        "feature_importance": dict(zip(

            rf_model.feature_names_in_,

            rf_model.feature_importances_

        ))
    })

# ==========================================================
# Run Server
# ==========================================================

if __name__ == "__main__":

    print("\nHemoScan AI running at http://localhost:5000\n")

    app.run(debug=True)