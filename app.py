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
import resend

from datetime import datetime
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
# Configure Resend
# ==========================================================

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
resend.api_key = RESEND_API_KEY

if RESEND_API_KEY:
    print("✓ Resend configured")
else:
    print("⚠ RESEND_API_KEY not set — /send-report will be unavailable")

# ==========================================================
# Helper Functions
# ==========================================================

def encode_gender(gender):
    gender = gender.lower()
    if gender == "female":
        return 0
    else:
        return 1


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

        "is_anemia": bool(is_anemia),

        "confidence": round(confidence, 2),

        "message": message,

        "model_used": type(rf_model).__name__,

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
# Email Helpers
# ==========================================================

def param_badge(value, low, high) -> str:
    """Returns a coloured HTML status badge for the email table."""
    try:
        v = float(value)
        if v < low:  return '<span style="color:#e8472a;font-weight:600;">Low</span>'
        if v > high: return '<span style="color:#e8a12a;font-weight:600;">High</span>'
        return              '<span style="color:#2adb7a;font-weight:600;">Normal</span>'
    except (ValueError, TypeError):
        return "—"


def build_email_html(prediction, is_anemia, confidence,
                     message, model_used, params) -> str:
    """
    Builds a fully inline-styled HTML email body compatible with
    Gmail, Outlook, and Apple Mail.
    """
    result_color = "#e8472a" if is_anemia else "#2adb7a"
    result_bg    = "#fff2f0" if is_anemia else "#f0fff6"
    border_left  = "#e8472a" if is_anemia else "#2adb7a"
    emoji        = "🩸"      if is_anemia else "✅"

    gender     = str(params.get("gender", "—")).capitalize()
    hemoglobin = params.get("hemoglobin", "—")
    mcv        = params.get("mcv",  "—")
    mch        = params.get("mch",  "—")
    mchc       = params.get("mchc", "—")
    generated  = datetime.now().strftime("%d %b %Y, %I:%M %p")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>HemoScan AI — Prediction Report</title>
</head>
<body style="margin:0;padding:0;background:#f0f0f0;font-family:'Segoe UI',Arial,sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0"
       style="background:#f0f0f0;padding:40px 16px;">
  <tr><td align="center">
  <table width="600" cellpadding="0" cellspacing="0"
         style="max-width:600px;background:#ffffff;border-radius:14px;
                overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.12);">

    <!-- ── HEADER ────────────────────────────────────── -->
    <tr>
      <td style="background:#0d0e10;padding:28px 36px;">
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td>
              <div style="font-family:'Georgia',serif;font-size:1.55rem;
                          font-weight:700;color:#f0ede8;letter-spacing:-0.02em;">
                HemoScan
                <span style="color:#e8472a;font-style:italic;"> AI</span>
              </div>
              <div style="font-family:'Courier New',monospace;font-size:0.70rem;
                          color:#5c5751;letter-spacing:0.10em;margin-top:4px;">
                ANEMIA PREDICTION REPORT
              </div>
            </td>
            <td align="right">
              <div style="font-family:'Courier New',monospace;font-size:0.70rem;
                          color:#5c5751;letter-spacing:0.06em;">
                {generated}
              </div>
            </td>
          </tr>
        </table>
      </td>
    </tr>

    <!-- ── VERDICT BANNER ────────────────────────────── -->
    <tr>
      <td style="background:{result_bg};padding:28px 36px;
                 border-left:5px solid {border_left};">
        <div style="font-size:2rem;margin-bottom:8px;">
          {emoji}
          <span style="font-family:'Georgia',serif;font-weight:700;
                       color:{result_color};font-size:1.55rem;
                       vertical-align:middle;">
            {prediction}
          </span>
        </div>
        <div style="font-size:1rem;color:#555;margin-bottom:10px;">
          Model confidence:
          <strong style="color:{result_color};font-size:1.1rem;">
            {confidence}%
          </strong>
        </div>
        <div style="font-size:0.90rem;color:#444;line-height:1.70;">
          {message}
        </div>
        <div style="margin-top:12px;font-family:'Courier New',monospace;
                    font-size:0.70rem;color:#999;
                    background:rgba(0,0,0,0.04);display:inline-block;
                    padding:4px 10px;border-radius:4px;">
          🌲 {model_used}
        </div>
      </td>
    </tr>

    <!-- ── CBC TABLE ─────────────────────────────────── -->
    <tr>
      <td style="padding:28px 36px;">
        <div style="font-size:0.95rem;font-weight:700;color:#111;
                    letter-spacing:-0.01em;margin-bottom:14px;
                    padding-bottom:8px;border-bottom:2px solid #f0f0f0;">
          CBC Parameter Analysis
        </div>
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;font-size:0.88rem;">
          <thead>
            <tr style="background:#f8f8f8;">
              <th style="padding:10px 14px;border:1px solid #e8e8e8;
                         text-align:left;font-weight:600;color:#333;
                         font-size:0.82rem;text-transform:uppercase;
                         letter-spacing:0.04em;">
                Parameter
              </th>
              <th style="padding:10px 14px;border:1px solid #e8e8e8;
                         text-align:left;font-weight:600;color:#333;
                         font-size:0.82rem;text-transform:uppercase;
                         letter-spacing:0.04em;">
                Value
              </th>
              <th style="padding:10px 14px;border:1px solid #e8e8e8;
                         text-align:left;font-weight:600;color:#333;
                         font-size:0.82rem;text-transform:uppercase;
                         letter-spacing:0.04em;">
                Status
              </th>
              <th style="padding:10px 14px;border:1px solid #e8e8e8;
                         text-align:left;font-weight:600;color:#333;
                         font-size:0.82rem;text-transform:uppercase;
                         letter-spacing:0.04em;">
                Normal Range
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#333;">
                Biological Sex
              </td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;font-weight:600;color:#333;">
                {gender}
              </td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;">—</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#999;font-size:0.82rem;">—</td>
            </tr>
            <tr style="background:#fafafa;">
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#333;">Hemoglobin</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;font-weight:600;color:#333;">{hemoglobin} g/dL</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;">{param_badge(hemoglobin, 12, 17)}</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#999;font-size:0.82rem;">12 – 17 g/dL</td>
            </tr>
            <tr>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#333;">MCV</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;font-weight:600;color:#333;">{mcv} fL</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;">{param_badge(mcv, 80, 100)}</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#999;font-size:0.82rem;">80 – 100 fL</td>
            </tr>
            <tr style="background:#fafafa;">
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#333;">MCH</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;font-weight:600;color:#333;">{mch} pg</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;">{param_badge(mch, 27, 33)}</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#999;font-size:0.82rem;">27 – 33 pg</td>
            </tr>
            <tr>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#333;">MCHC</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;font-weight:600;color:#333;">{mchc} g/dL</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;">{param_badge(mchc, 32, 36)}</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#999;font-size:0.82rem;">32 – 36 g/dL</td>
            </tr>
          </tbody>
        </table>
      </td>
    </tr>

    

    <!-- ── FOOTER ─────────────────────────────────────── -->
    <tr>
      <td style="background:#f8f8f8;padding:16px 36px;
                 border-top:1px solid #eee;text-align:center;">
        <div style="font-family:'Courier New',monospace;font-size:0.72rem;color:#bbb;">
          HemoScan AI &nbsp;·&nbsp; Generated {generated}
        </div>
        <div style="font-family:'Courier New',monospace;font-size:0.68rem;
                    color:#ccc;margin-top:4px;">
          This is an automated report. Do not reply to this email.
        </div>
      </td>
    </tr>

  </table>
  </td></tr>
</table>

</body>
</html>"""
# ==========================================================
# CSV Report Export
# ==========================================================

import csv

REPORT_CSV_PATH = "last_report.csv"

def save_report_as_csv(prediction, is_anemia, confidence,
                       message, model_used, params) -> bool:
    """
    Saves the exported report values as a CSV file
    (last_report.csv), in parallel with sending the email.

    Each row contains: field, value, unit, status, normal_range.
    A header row is always written so the file is self-describing.
    New reports are appended as rows so history is preserved.
    """
    generated  = datetime.now().strftime("%d %b %Y, %I:%M %p")
    gender     = str(params.get("gender",     "—")).capitalize()
    hemoglobin = params.get("hemoglobin", "—")
    mcv        = params.get("mcv",        "—")
    mch        = params.get("mch",        "—")
    mchc       = params.get("mchc",       "—")

    def param_status(value, low, high):
        try:
            v = float(value)
            if v < low:  return "Low"
            if v > high: return "High"
            return "Normal"
        except (ValueError, TypeError):
            return "Unknown"

    # Each row: timestamp, model, prediction, confidence, gender,
    #           hgb, hgb_status, mcv, mcv_status,
    #           mch, mch_status, mchc, mchc_status
    fieldnames = [
        "Generated", "Model", "Prediction", "Is_Anemia", "Confidence_%", "Message",
        "Gender",
        "Hemoglobin_g_dL", "Hemoglobin_Status", "Hemoglobin_Range",
        "MCV_fL",          "MCV_Status",         "MCV_Range",
        "MCH_pg",          "MCH_Status",          "MCH_Range",
        "MCHC_g_dL",       "MCHC_Status",         "MCHC_Range",
    ]

    row = {
        "Generated":          generated,
        "Model":              model_used,
        "Prediction":         prediction,
        "Is_Anemia":          1 if is_anemia else 0,
        "Confidence_%":       confidence,
        "Message":            message,
        "Gender":             gender,
        "Hemoglobin_g_dL":    hemoglobin,
        "Hemoglobin_Status":  param_status(hemoglobin, 12, 17),
        "Hemoglobin_Range":   "12 – 17 g/dL",
        "MCV_fL":             mcv,
        "MCV_Status":         param_status(mcv, 80, 100),
        "MCV_Range":          "80 – 100 fL",
        "MCH_pg":             mch,
        "MCH_Status":         param_status(mch, 27, 33),
        "MCH_Range":          "27 – 33 pg",
        "MCHC_g_dL":          mchc,
        "MCHC_Status":        param_status(mchc, 32, 36),
        "MCHC_Range":         "32 – 36 g/dL",
    }

    try:
        file_exists = os.path.isfile(REPORT_CSV_PATH)
        with open(REPORT_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()   # write header only once
            writer.writerow(row)
        print(f"✓ Report CSV saved → {REPORT_CSV_PATH}")
        return True
    except Exception as exc:
        print(f"⚠ Could not save report CSV: {exc}")
        return False


# ── POST /send-report ─────────────────────────────────────────────────────
@app.route("/send-report", methods=["POST"])
def send_report():
    """
    Build a styled HTML email from prediction data and send it
    to the specified address using the Resend API.

    Request JSON:
    {
        "to_email":   "patient@example.com",
        "prediction": "Anemia Detected",
        "is_anemia":  true,
        "confidence": 94.0,
        "message":    "...",
        "model_used": "RandomForestClassifier",
        "params": {
            "gender":     "female",
            "hemoglobin": 9.5,
            "mcv":        68.0,
            "mch":        22.0,
            "mchc":       30.0
        }
    }

    Response JSON:
        { "success": true, "message": "Report sent to patient@example.com",
          "resend_id": "..." }
    """
    if not RESEND_API_KEY:
        return jsonify({
            "error": "Resend is not configured. "
                     "Set the RESEND_API_KEY environment variable and restart."
        }), 503

    body       = request.get_json(force=True) or {}
    to_email   = body.get("to_email",   "").strip()
    prediction = body.get("prediction", "")
    is_anemia  = bool(body.get("is_anemia", False))
    confidence = body.get("confidence", 0)
    message    = body.get("message",    "")
    model_used = body.get("model_used", "RandomForestClassifier")
    params     = body.get("params",     {})

    # Validate email
    if not to_email or not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", to_email):
        return jsonify({"error": "Please provide a valid email address."}), 400

    # Build HTML email
    html = build_email_html(
        prediction, is_anemia, confidence,
        message, model_used, params
    )

    subject = (f"HemoScan AI — {prediction} "
               f"({confidence}% confidence)")

    try:
        # ── Save report values to CSV in parallel ──────────────────
        csv_saved = save_report_as_csv(
            prediction, is_anemia, confidence,
            message, model_used, params
        )

        result = resend.Emails.send({
            # During Resend free-tier testing you can only send FROM
            # onboarding@resend.dev unless you verify your own domain.
            # Once you verify a domain, change this to your own address.
            "from":    "HemoScan AI <onboarding@resend.dev>",
            "to":      [to_email],
            "subject": subject,
            "html":    html,
        })

        return jsonify({
            "success":   True,
            "message":   f"Report sent to {to_email}",
            "resend_id": result.get("id", ""),
            "csv_saved": csv_saved,
            "csv_file":  REPORT_CSV_PATH if csv_saved else None
        })

    except Exception as exc:
        # Resend raises exceptions for API errors (bad key, invalid address, etc.)
        return jsonify({"error": str(exc)}), 500


# ==========================================================
# Run Server
# ==========================================================

if __name__ == "__main__":

    print("\nHemoScan AI running at http://localhost:5000\n")

    app.run(debug=True)