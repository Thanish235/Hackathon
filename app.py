# SECTION 1 — Imports


import os
import re
import csv
import json
import pickle
import warnings

import pytesseract
import pdfplumber
import pandas as pd
import requests

from datetime import datetime
from PIL      import Image
from groq     import Groq
from flask    import Flask, request, jsonify, send_from_directory
from dotenv   import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — Flask App Initialisation
# ══════════════════════════════════════════════════════════════════════

# Serve static files (index.html, styles.css, script.js) from the
# same directory as this file.
app = Flask(__name__, static_folder=".", static_url_path="")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — ML Model Loading  (RandomForestClassifier)
# ══════════════════════════════════════════════════════════════════════

MODEL_PATH = "model5-2.pkl"

try:
    with open(MODEL_PATH, "rb") as _f:
        rf_model = pickle.load(_f)
    print("✓ Model loaded:", rf_model.feature_names_in_)
except Exception as _e:
    rf_model = None
    print("✗ Model loading failed:", _e)


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — Groq Client Configuration  (AI extraction + treatment)
# ══════════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✓ Groq configured")
else:
    groq_client = None
    print("⚠  GROQ_API_KEY not set — /extract and /treatment will be unavailable")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — Brevo Email Configuration
#=
# ══════════════════════════════════════════════════════════════════════

BREVO_API_KEY      = os.getenv("BREVO_API_KEY")
BREVO_SENDER_EMAIL = os.getenv("BREVO_SENDER_EMAIL", "hemoscan@yourdomain.com")
BREVO_SENDER_NAME  = os.getenv("BREVO_SENDER_NAME",  "HemoScan AI")

if BREVO_API_KEY:
    print("✓ Brevo configured")
else:
    print("⚠  BREVO_API_KEY not set — /send-report will be unavailable")


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — CSV Report Storage Path
# ══════════════════════════════════════════════════════════════════════

# Every sent or exported report is appended here for history tracking.
REPORT_CSV_PATH = "last_report.csv"


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — ML Helper Utilities
# ══════════════════════════════════════════════════════════════════════

def encode_gender(gender: str) -> int:
    """Encode gender string to binary integer for the model.
    Female → 0  |  Male / Other → 1
    """
    return 0 if gender.lower() == "female" else 1


def anemia_subtype(mcv: float, mch: float) -> str:
    if mcv < 80 and mch < 27:
        return "microcytic hypochromic anemia"
    if mcv > 100:
        return "macrocytic anemia"
    return "normocytic anemia"


def build_features(hb: float, mcv: float, mch: float,
                   mchc: float, g: int) -> "pd.DataFrame":
    """Build a single-row DataFrame matching the model's expected feature order.

    New model (model5-2.pkl) trained on BALANCED_ANEMIA_DATASET.csv expects:
    ['Hemoglobin', 'MCV', 'MCH', 'MCHC', 'Gender']
    Dataset: 65,796 balanced samples | Accuracy: 99.97%
    """
    return pd.DataFrame(
        [[hb, mcv, mch, mchc, g]],
        columns=["Hemoglobin", "MCV", "MCH", "MCHC", "Gender"]
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — Text Extraction from Uploaded Files
# ══════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file) -> str:
    """Extract all text from every page of an uploaded PDF."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_image(file) -> str:
    """Run Tesseract OCR on an uploaded image and return raw text."""
    img  = Image.open(file).convert("RGB")
    return pytesseract.image_to_string(img)


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — Groq: CBC Value Extraction from Raw Text
# ══════════════════════════════════════════════════════════════════════

def extract_cbc_with_groq(text: str) -> dict:
    prompt = f"""Extract these CBC values from the medical report.

Return ONLY valid JSON — no explanation, no markdown fences.

Format:
{{
  "hemoglobin": number or null,
  "mcv":        number or null,
  "mch":        number or null,
  "mchc":       number or null
}}

Report:
{text}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(raw)
    except Exception:
        return {"hemoglobin": None, "mcv": None, "mch": None, "mchc": None}


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — Risk Severity Score Calculator
#
#   Weighted 0–100 composite score reflecting how far CBC values
#   deviate from normal ranges. Higher = worse.
#
#   Weights (sum = 1.0):
#     Hemoglobin 40% | MCV 25% | MCH 20% | MCHC 15%
#
#   Levels:  0–19 Minimal | 20–44 Mild | 45–69 Moderate | 70+ Severe
# ══════════════════════════════════════════════════════════════════════

def compute_severity_score(hb: float, mcv: float, mch: float,
                           mchc: float, gender_str: str) -> dict:
    """Compute a composite 0–100 risk severity score.

    Returns:
        score (float), level (str), label (str), per_param (dict)
    """
    is_female      = gender_str.lower() == "female"
    hb_min, hb_max = (12.0, 16.0) if is_female else (13.5, 17.5)

    # Tuple: (value, low, high, weight)
    ranges = {
        "Hemoglobin": (hb,   hb_min, hb_max,  0.40),
        "MCV":        (mcv,  80.0,   100.0,   0.25),
        "MCH":        (mch,  27.0,   33.0,    0.20),
        "MCHC":       (mchc, 32.0,   36.0,    0.15),
    }

    weighted_sum = 0.0
    per_param    = {}

    for name, (val, lo, hi, weight) in ranges.items():
        span      = hi - lo
        excess    = max(0.0, lo - val) + max(0.0, val - hi)
        deviation = min(excess / (span * 0.5), 1.0)   # 0–1, capped at 1
        per_param[name]  = round(deviation * 100, 1)
        weighted_sum    += deviation * weight

    score = round(min(weighted_sum * 100, 100), 1)

    if score < 20:   level, label = "minimal",  "Minimal Risk"
    elif score < 45: level, label = "mild",     "Mild"
    elif score < 70: level, label = "moderate", "Moderate"
    else:            level, label = "severe",   "Severe"

    return {"score": score, "level": level, "label": label, "per_param": per_param}


# ══════════════════════════════════════════════════════════════════════
# SECTION 11 — SHAP-style Instance-Level Feature Importances
#
#   Combines the RF global feature_importances_ with each parameter's
#   deviation from its normal centre to produce instance-level
#   attribution (how much did THIS patient's values drive the result).
#
#   instance_weight = global_importance × (0.5 + 0.5 × deviation)
#   Then normalised so all contributions sum to 100%.
# ══════════════════════════════════════════════════════════════════════

def compute_shap_importances(hb: float, mcv: float, mch: float,
                             mchc: float, gender_enc: int) -> list:
    """Return per-feature SHAP-style contribution list, sorted descending.

    Each entry: { feature, pct, deviation, direction }
    """
    if rf_model is None:
        return []

    global_imp = dict(zip(
        rf_model.feature_names_in_,
        rf_model.feature_importances_
    ))

    # (value, normal_centre, half_span)
    param_meta = {
        "Hemoglobin": (hb,          14.5,  2.5),
        "MCV":        (mcv,         90.0,  10.0),
        "MCH":        (mch,         30.0,  3.0),
        "MCHC":       (mchc,        34.0,  2.0),
        "Gender":     (gender_enc,  0.5,   0.5),
    }

    results = []
    for feat, g_imp in global_imp.items():
        meta = param_meta.get(feat)
        if not meta:
            continue
        val, centre, span = meta
        deviation       = abs(val - centre) / span
        instance_weight = g_imp * (0.5 + 0.5 * deviation)
        results.append({
            "feature":    feat,
            "importance": round(instance_weight, 4),
            "deviation":  round(deviation * 100, 1),
            "direction":  "high" if val > centre else "low",
        })

    # Normalise contributions to sum to 100%
    total = sum(r["importance"] for r in results) or 1
    for r in results:
        r["pct"] = round(r["importance"] / total * 100, 1)

    results.sort(key=lambda x: x["pct"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════
# SECTION 12 — Groq: AI Treatment & Dietary Suggestions
#
#   Sends full patient context to Groq Llama 3. Returns structured
#   JSON: summary, diet, supplements, lifestyle, when_to_see_doctor,
#   disclaimer — all personalised to the detected subtype + severity.
# ══════════════════════════════════════════════════════════════════════

def get_treatment_suggestions(prediction: str, is_anemia: bool, subtype: str,
                              hb: float, mcv: float, mch: float, mchc: float,
                              gender_str: str, severity_score: float) -> dict:
    """Call Groq to generate personalised clinical treatment guidance."""
    if groq_client is None:
        return {"error": "Groq not configured"}

    prompt = f"""You are a clinical decision-support assistant.
A patient's CBC has been analysed by an AI anemia-detection model.

Patient data:
  Gender        : {gender_str}
  Hemoglobin    : {hb} g/dL   (normal 12–17)
  MCV           : {mcv} fL    (normal 80–100)
  MCH           : {mch} pg    (normal 27–33)
  MCHC          : {mchc} g/dL (normal 32–36)
  Prediction    : {prediction}
  Anemia type   : {subtype}
  Severity score: {severity_score}/100

Return ONLY valid JSON — no markdown, no explanation:
{{
  "summary":            "2-sentence plain-English summary for the patient",
  "diet":               ["item 1", "item 2", "item 3", "item 4"],
  "supplements":        ["supplement + dose 1", "supplement + dose 2"],
  "lifestyle":          ["tip 1", "tip 2", "tip 3"],
  "when_to_see_doctor": "one clear sentence on urgency",
  "disclaimer":         "one sentence medical disclaimer"
}}

Be specific to the anemia subtype. Keep each list item under 12 words.
"""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as exc:
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════
# SECTION 13 — Email / Report HTML Section Builders
#
#   Each helper returns an inline-styled <tr> block compatible with
#   Gmail, Outlook, Apple Mail, and browser print.
#   All called from build_report_html() to assemble the full document.
# ══════════════════════════════════════════════════════════════════════

def _email_param_badge(value, low: float, high: float) -> str:
    """Return a coloured HTML status badge: Low / Normal / High."""
    try:
        v = float(value)
        if v < low:  return '<span style="color:#e8472a;font-weight:600;">Low</span>'
        if v > high: return '<span style="color:#e8a12a;font-weight:600;">High</span>'
        return               '<span style="color:#2adb7a;font-weight:600;">Normal</span>'
    except (ValueError, TypeError):
        return "—"


def _email_severity_section(severity: dict) -> str:
    """Build the Risk Severity Score section (circular badge + param bars)."""
    if not severity:
        return ""

    score     = severity.get("score", 0)
    label     = severity.get("label", "—")
    level     = severity.get("level", "minimal")
    per_param = severity.get("per_param", {})

    colour = {"minimal": "#2adb7a", "mild": "#f0b429",
              "moderate": "#e8a12a", "severe": "#e8472a"}.get(level, "#e8472a")

    bar_rows = ""
    for key, pct in per_param.items():
        c = "#e8472a" if pct > 60 else "#f0b429" if pct > 30 else "#2adb7a"
        bar_rows += (
            f'<tr><td style="padding:4px 0;font-family:\'Courier New\',monospace;'
            f'font-size:0.78rem;color:#555;width:90px;">{key}</td>'
            f'<td style="padding:4px 8px;"><table width="100%" cellpadding="0"'
            f' cellspacing="0"><tr><td style="background:#f0f0f0;border-radius:4px;'
            f'overflow:hidden;height:8px;"><div style="width:{pct}%;height:8px;'
            f'background:{c};border-radius:4px;"></div></td></tr></table></td>'
            f'<td style="padding:4px 0;font-family:\'Courier New\',monospace;'
            f'font-size:0.75rem;color:{c};width:38px;text-align:right;">'
            f'{pct}%</td></tr>'
        )

    return f"""
    <tr>
      <td style="padding:24px 36px;border-top:1px solid #f0f0f0;">
        <div style="font-size:0.95rem;font-weight:700;color:#111;margin-bottom:16px;
                    padding-bottom:8px;border-bottom:2px solid #f0f0f0;">
          ⚠️ Risk Severity Score
        </div>
        <table width="100%" cellpadding="0" cellspacing="0"><tr>
          <td style="width:130px;text-align:center;vertical-align:middle;padding-right:24px;">
            <div style="display:inline-block;width:100px;height:100px;border-radius:50%;
                        border:6px solid {colour};line-height:88px;text-align:center;">
              <span style="font-family:'Courier New',monospace;font-size:1.8rem;
                           font-weight:700;color:{colour};">{score}</span>
            </div>
            <div style="font-family:'Courier New',monospace;font-size:0.72rem;
                        color:{colour};margin-top:6px;text-align:center;
                        text-transform:uppercase;">{label}</div>
          </td>
          <td style="vertical-align:middle;">
            <table width="100%" cellpadding="0" cellspacing="0">{bar_rows}</table>
          </td>
        </tr></table>
      </td>
    </tr>"""


def _email_shap_section(shap: list) -> str:
    """Build the Prediction Explainability bar-chart section.

    Red = value above normal centre | Green = value below normal centre.
    """
    if not shap:
        return ""

    rows = ""
    for item in shap:
        feat   = item.get("feature", "")
        pct    = item.get("pct", 0)
        direc  = item.get("direction", "high")
        dev    = item.get("deviation", 0)
        colour = "#e8472a" if direc == "high" else "#2adb7a"
        arrow  = "↑" if direc == "high" else "↓"
        rows += (
            f'<tr style="border-bottom:1px solid #f8f8f8;">'
            f'<td style="padding:8px 0;font-family:\'Courier New\',monospace;'
            f'font-size:0.80rem;color:#555;width:100px;">{feat}</td>'
            f'<td style="padding:8px 12px;"><table width="100%" cellpadding="0"'
            f' cellspacing="0"><tr><td style="background:#f0f0f0;border-radius:4px;'
            f'overflow:hidden;height:10px;"><div style="width:{pct}%;height:10px;'
            f'background:{colour};border-radius:4px;"></div></td></tr></table></td>'
            f'<td style="padding:8px 0;font-family:\'Courier New\',monospace;'
            f'font-size:0.78rem;font-weight:600;color:{colour};width:38px;'
            f'text-align:right;">{pct}%</td>'
            f'<td style="padding:8px 0 8px 12px;font-family:\'Courier New\',monospace;'
            f'font-size:0.72rem;color:#999;width:110px;">{arrow} {dev}% deviation</td>'
            f'</tr>'
        )

    return f"""
    <tr>
      <td style="padding:24px 36px;border-top:1px solid #f0f0f0;background:#fafafa;">
        <div style="font-size:0.95rem;font-weight:700;color:#111;margin-bottom:16px;
                    padding-bottom:8px;border-bottom:2px solid #f0f0f0;">
          🔍 Prediction Explainability
        </div>
        <div style="font-family:'Courier New',monospace;font-size:0.72rem;
                    color:#999;margin-bottom:12px;">
          Contribution of each parameter to this prediction
        </div>
        <table width="100%" cellpadding="0" cellspacing="0">{rows}</table>
      </td>
    </tr>"""


def _email_treatment_section(treatment: dict, is_anemia: bool) -> str:
    """Build the AI Treatment Suggestions section (3-column card layout)."""
    if not treatment or treatment.get("error"):
        return ""

    summary    = treatment.get("summary",            "")
    diet       = treatment.get("diet",               [])
    supps      = treatment.get("supplements",        [])
    lifestyle  = treatment.get("lifestyle",          [])
    doctor     = treatment.get("when_to_see_doctor", "")
    disclaimer = treatment.get("disclaimer",         "")
    result_col = "#e8472a" if is_anemia else "#2adb7a"

    def _list_html(items: list) -> str:
        if not items:
            return ""
        lis = "".join(
            f'<li style="padding:3px 0;font-size:0.83rem;color:#444;'
            f'line-height:1.5;">{item}</li>' for item in items
        )
        return f'<ul style="margin:6px 0 0 16px;padding:0;">{lis}</ul>'

    def _card(title: str, emoji: str, items: list) -> str:
        if not items:
            return ""
        return (
            '<td style="width:33%;vertical-align:top;padding:0 6px;">'
            '<div style="background:#fff;border:1px solid #ebebeb;border-radius:10px;'
            f'padding:14px 16px;min-height:120px;">'
            f'<div style="font-size:0.85rem;font-weight:700;color:#111;'
            f'margin-bottom:6px;">{emoji} {title}</div>'
            + _list_html(items) + '</div></td>'
        )

    summary_html = (
        '<div style="font-size:0.88rem;color:#444;line-height:1.65;padding:10px 14px;'
        f'background:#f9f9f9;border-left:3px solid {result_col};">'
        + summary + '</div><br/>'
    ) if summary else ""

    doctor_html = (
        '<div style="background:#fff3f2;border:1px solid #ffd5d0;border-radius:8px;'
        'padding:10px 14px;font-size:0.83rem;color:#c0392b;margin-bottom:10px;">🏥 '
        + doctor + '</div>'
    ) if doctor else ""

    disclaimer_html = (
        '<div style="font-family:monospace;font-size:0.70rem;color:#bbb;'
        'margin-top:8px;">' + disclaimer + '</div>'
    ) if disclaimer else ""

    return (
        '\n    <tr>\n'
        '      <td style="padding:24px 36px;border-top:1px solid #f0f0f0;">\n'
        '        <div style="font-size:0.95rem;font-weight:700;color:#111;'
        'margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid #f0f0f0;">'
        '💊 AI Treatment Suggestions</div>\n'
        + summary_html
        + '\n        <table width="100%" cellpadding="0" cellspacing="0"'
        ' style="margin-bottom:14px;"><tr>'
        + _card("Diet", "🥗", diet)
        + _card("Supplements", "💊", supps)
        + _card("Lifestyle", "🏃", lifestyle)
        + '</tr></table>\n'
        + doctor_html + disclaimer_html
        + '\n      </td>\n    </tr>'
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 14 — Full HTML Report Builder
#
#   Builds one complete, self-contained, inline-styled HTML document
#   used by BOTH /send-report (email) AND /export-report (print/PDF).
#
#   Report sections (in order):
#     1. Header            — branding + timestamp
#     2. Verdict banner    — prediction + confidence + model tag
#     3. CBC table         — values, status badges, normal ranges
#     4. Severity score    — circular badge + per-param deviation bars
#     5. Explainability    — SHAP-style feature contribution bars
#     6. Treatment advice  — diet / supplements / lifestyle cards
#     7. Footer            — clinical disclaimer + generation time
# ══════════════════════════════════════════════════════════════════════

def build_report_html(prediction: str, is_anemia: bool, confidence: float,
                      message: str, model_used: str, params: dict,
                      severity=None, shap=None, treatment=None) -> str:
    """Assemble the complete HTML prediction report.

    Shared by both the email sender and the print/export endpoint.
    Optional sections (severity, shap, treatment) are omitted when None.
    """
    result_color = "#e8472a" if is_anemia else "#2adb7a"
    result_bg    = "#fff2f0" if is_anemia else "#f0fff6"
    border_left  = "#e8472a" if is_anemia else "#2adb7a"
    emoji        = "🩸"      if is_anemia else "✅"

    gender     = str(params.get("gender",     "—")).capitalize()
    hemoglobin = params.get("hemoglobin", "—")
    mcv        = params.get("mcv",        "—")
    mch        = params.get("mch",        "—")
    mchc       = params.get("mchc",       "—")
    generated  = datetime.now().strftime("%d %b %Y, %I:%M %p")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>HemoScan AI — Prediction Report</title>
  <style>
    /* Force colour printing and hide no-print elements */
    @media print {{
      body  {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
      .no-print {{ display: none !important; }}
    }}
  </style>
</head>
<body style="margin:0;padding:0;background:#f0f0f0;
             font-family:'Segoe UI',Arial,sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0"
       style="background:#f0f0f0;padding:40px 16px;">
  <tr><td align="center">
  <table width="600" cellpadding="0" cellspacing="0"
         style="max-width:600px;background:#ffffff;border-radius:14px;
                overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.12);">

    <!-- ══ 1. HEADER ════════════════════════════════════════ -->
    <tr>
      <td style="background:#0d0e10;padding:28px 36px;">
        <table width="100%" cellpadding="0" cellspacing="0"><tr>
          <td>
            <div style="font-family:'Georgia',serif;font-size:1.55rem;
                        font-weight:700;color:#f0ede8;letter-spacing:-0.02em;">
              HemoScan<span style="color:#e8472a;font-style:italic;"> AI</span>
            </div>
            <div style="font-family:'Courier New',monospace;font-size:0.70rem;
                        color:#5c5751;letter-spacing:0.10em;margin-top:4px;">
              ANEMIA PREDICTION REPORT
            </div>
          </td>
          <td align="right">
            <div style="font-family:'Courier New',monospace;font-size:0.70rem;
                        color:#5c5751;">{generated}</div>
          </td>
        </tr></table>
      </td>
    </tr>

    <!-- ══ 2. VERDICT BANNER ═════════════════════════════════ -->
    <tr>
      <td style="background:{result_bg};padding:28px 36px;
                 border-left:5px solid {border_left};">
        <div style="font-size:2rem;margin-bottom:8px;">
          {emoji}
          <span style="font-family:'Georgia',serif;font-weight:700;
                       color:{result_color};font-size:1.55rem;
                       vertical-align:middle;">{prediction}</span>
        </div>
        <div style="font-size:1rem;color:#555;margin-bottom:10px;">
          Model confidence:
          <strong style="color:{result_color};font-size:1.1rem;">{confidence}%</strong>
        </div>
        <div style="font-size:0.90rem;color:#444;line-height:1.70;">{message}</div>
        <div style="margin-top:12px;font-family:'Courier New',monospace;font-size:0.70rem;
                    color:#999;background:rgba(0,0,0,0.04);display:inline-block;
                    padding:4px 10px;border-radius:4px;">🌲 {model_used}</div>
      </td>
    </tr>

    <!-- ══ 3. CBC PARAMETER TABLE ═════════════════════════════ -->
    <tr>
      <td style="padding:28px 36px;">
        <div style="font-size:0.95rem;font-weight:700;color:#111;margin-bottom:14px;
                    padding-bottom:8px;border-bottom:2px solid #f0f0f0;">
          CBC Parameter Analysis
        </div>
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;font-size:0.88rem;">
          <thead>
            <tr style="background:#f8f8f8;">
              <th style="padding:10px 14px;border:1px solid #e8e8e8;text-align:left;
                         font-weight:600;color:#333;font-size:0.82rem;
                         text-transform:uppercase;letter-spacing:0.04em;">Parameter</th>
              <th style="padding:10px 14px;border:1px solid #e8e8e8;text-align:left;
                         font-weight:600;color:#333;font-size:0.82rem;
                         text-transform:uppercase;letter-spacing:0.04em;">Value</th>
              <th style="padding:10px 14px;border:1px solid #e8e8e8;text-align:left;
                         font-weight:600;color:#333;font-size:0.82rem;
                         text-transform:uppercase;letter-spacing:0.04em;">Status</th>
              <th style="padding:10px 14px;border:1px solid #e8e8e8;text-align:left;
                         font-weight:600;color:#333;font-size:0.82rem;
                         text-transform:uppercase;letter-spacing:0.04em;">Range</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#333;">Biological Sex</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;font-weight:600;color:#333;">{gender}</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;">—</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#999;font-size:0.82rem;">—</td>
            </tr>
            <tr style="background:#fafafa;">
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#333;">Hemoglobin</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;font-weight:600;color:#333;">{hemoglobin} g/dL</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;">{_email_param_badge(hemoglobin, 12, 17)}</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#999;font-size:0.82rem;">12–17 g/dL</td>
            </tr>
            <tr>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#333;">MCV</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;font-weight:600;color:#333;">{mcv} fL</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;">{_email_param_badge(mcv, 80, 100)}</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#999;font-size:0.82rem;">80–100 fL</td>
            </tr>
            <tr style="background:#fafafa;">
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#333;">MCH</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;font-weight:600;color:#333;">{mch} pg</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;">{_email_param_badge(mch, 27, 33)}</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#999;font-size:0.82rem;">27–33 pg</td>
            </tr>
            <tr>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#333;">MCHC</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;font-weight:600;color:#333;">{mchc} g/dL</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;">{_email_param_badge(mchc, 32, 36)}</td>
              <td style="padding:10px 14px;border:1px solid #e8e8e8;color:#999;font-size:0.82rem;">32–36 g/dL</td>
            </tr>
          </tbody>
        </table>
      </td>
    </tr>

    <!-- ══ 4. RISK SEVERITY SCORE ═════════════════════════════ -->
    {_email_severity_section(severity)}

    <!-- ══ 5. PREDICTION EXPLAINABILITY ══════════════════════ -->
    {_email_shap_section(shap)}

    <!-- ══ 6. AI TREATMENT SUGGESTIONS ══════════════════════ -->
    {_email_treatment_section(treatment, is_anemia)}

    <!-- ══ 7. FOOTER ════════════════════════════════════════ -->
    <tr>
      <td style="background:#f8f8f8;padding:16px 36px;
                 border-top:1px solid #eee;text-align:center;">
        <div style="font-family:'Courier New',monospace;font-size:0.72rem;color:#bbb;">
          HemoScan AI &nbsp;·&nbsp; Generated {generated}
        </div>
        <div style="font-family:'Courier New',monospace;font-size:0.68rem;
                    color:#ccc;margin-top:4px;">
          For clinical decision support only. Not a substitute for professional
          medical advice. Please consult a qualified healthcare provider.
        </div>
      </td>
    </tr>

  </table>
  </td></tr>
</table>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════
# SECTION 15 — CSV Report History Logger
#
#   Appends every sent or exported report as a new row in
#   last_report.csv. Header is written only on first file creation.
#   Builds a permanent history of all predictions over time.
# ══════════════════════════════════════════════════════════════════════

def save_report_as_csv(prediction: str, is_anemia: bool, confidence: float,
                       message: str, model_used: str, params: dict) -> bool:
    """Append this report's data to last_report.csv. Returns True on success."""
    generated  = datetime.now().strftime("%d %b %Y, %I:%M %p")
    gender     = str(params.get("gender",     "—")).capitalize()
    hemoglobin = params.get("hemoglobin", "—")
    mcv        = params.get("mcv",        "—")
    mch        = params.get("mch",        "—")
    mchc       = params.get("mchc",       "—")

    def _status(val, lo: float, hi: float) -> str:
        try:
            v = float(val)
            if v < lo:  return "Low"
            if v > hi:  return "High"
            return "Normal"
        except (ValueError, TypeError):
            return "Unknown"

    fieldnames = [
        "Generated", "Model", "Prediction", "Is_Anemia", "Confidence_%", "Message",
        "Gender",
        "Hemoglobin_g_dL", "Hemoglobin_Status", "Hemoglobin_Range",
        "MCV_fL",          "MCV_Status",         "MCV_Range",
        "MCH_pg",          "MCH_Status",          "MCH_Range",
        "MCHC_g_dL",       "MCHC_Status",         "MCHC_Range",
    ]
    row = {
        "Generated":         generated,
        "Model":             model_used,
        "Prediction":        prediction,
        "Is_Anemia":         1 if is_anemia else 0,
        "Confidence_%":      confidence,
        "Message":           message,
        "Gender":            gender,
        "Hemoglobin_g_dL":   hemoglobin,
        "Hemoglobin_Status": _status(hemoglobin, 12, 17),
        "Hemoglobin_Range":  "12–17 g/dL",
        "MCV_fL":            mcv,
        "MCV_Status":        _status(mcv, 80, 100),
        "MCV_Range":         "80–100 fL",
        "MCH_pg":            mch,
        "MCH_Status":        _status(mch, 27, 33),
        "MCH_Range":         "27–33 pg",
        "MCHC_g_dL":         mchc,
        "MCHC_Status":       _status(mchc, 32, 36),
        "MCHC_Range":        "32–36 g/dL",
    }

    try:
        file_exists = os.path.isfile(REPORT_CSV_PATH)
        with open(REPORT_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"✓ Report CSV saved → {REPORT_CSV_PATH}")
        return True
    except Exception as exc:
        print(f"⚠  Could not save report CSV: {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════
# SECTION 16 — Flask Routes
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main application page (index.html)."""
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Run the RandomForest model and return prediction + severity + SHAP.

    Request JSON:
        hemoglobin (float), mcv (float), mch (float),
        mchc (float), gender ("male"|"female")

    Response JSON:
        prediction, is_anemia, confidence, message, model_used,
        probabilities, severity, shap
    """
    if rf_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    try:
        hb         = float(data["hemoglobin"])
        mcv        = float(data["mcv"])
        mch        = float(data["mch"])
        mchc       = float(data["mchc"])
        gender_str = data.get("gender", "male")
        gender     = encode_gender(gender_str)
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid inputs — check all CBC fields"}), 400

    # Build feature matrix in new model's expected order:
    # ['Hemoglobin', 'MCV', 'MCH', 'MCHC', 'Gender']
    features   = build_features(hb, mcv, mch, mchc, gender)
    pred       = rf_model.predict(features)[0]
    proba      = rf_model.predict_proba(features)[0]
    # Classes are floats (0.0 / 1.0) in the new dataset
    is_anemia  = float(pred) == 1.0
    confidence = float(proba[1] if is_anemia else proba[0]) * 100

    if is_anemia:
        subtype = anemia_subtype(mcv, mch)
        message = f"Model predicts {subtype} with {confidence:.1f}% confidence."
    else:
        message = f"No anemia detected with {confidence:.1f}% confidence."

    return jsonify({
        "prediction":    "Anemia Detected" if is_anemia else "No Anemia",
        "is_anemia":     bool(is_anemia),
        "confidence":    round(confidence, 2),
        "message":       message,
        "model_used":    type(rf_model).__name__,
        "probabilities": {"no_anemia": float(proba[0]), "anemia": float(proba[1])},
        "severity":      compute_severity_score(hb, mcv, mch, mchc, gender_str),
        "shap":          compute_shap_importances(hb, mcv, mch, mchc, gender),
    })


@app.route("/extract", methods=["POST"])
def extract():
    """OCR/parse an uploaded CBC report and extract values using Groq.

    Accepts multipart/form-data with key 'file' (PDF / JPG / PNG).
    Returns: { success, hemoglobin, mcv, mch, mchc }
    """
    if groq_client is None:
        return jsonify({"error": "Groq API not configured"}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file     = request.files["file"]
    filename = file.filename.lower()

    try:
        text = (extract_text_from_pdf(file)
                if filename.endswith(".pdf")
                else extract_text_from_image(file))
        data = extract_cbc_with_groq(text)
        return jsonify({
            "success":    True,
            "hemoglobin": data.get("hemoglobin"),
            "mcv":        data.get("mcv"),
            "mch":        data.get("mch"),
            "mchc":       data.get("mchc"),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/model-info")
def model_info():
    """Return model metadata: name, features, classes, global importances."""
    if rf_model is None:
        return jsonify({"error": "Model not loaded"})
    return jsonify({
        "model":              type(rf_model).__name__,
        "features":           list(rf_model.feature_names_in_),
        "classes":            rf_model.classes_.tolist(),
        "feature_importance": dict(zip(
            rf_model.feature_names_in_,
            rf_model.feature_importances_
        )),
    })


@app.route("/treatment", methods=["POST"])
def treatment():
    """Compute severity + return AI treatment suggestions via Groq.

    Request JSON: prediction, is_anemia, gender,
                  hemoglobin, mcv, mch, mchc

    Response JSON: { severity, suggestions, subtype }
    """
    body       = request.get_json(force=True) or {}
    prediction = body.get("prediction", "")
    is_anemia  = bool(body.get("is_anemia", False))
    gender_str = body.get("gender",     "male")
    hb         = float(body.get("hemoglobin", 14))
    mcv        = float(body.get("mcv",   90))
    mch        = float(body.get("mch",   30))
    mchc       = float(body.get("mchc",  34))
    subtype    = anemia_subtype(mcv, mch) if is_anemia else "N/A"

    severity    = compute_severity_score(hb, mcv, mch, mchc, gender_str)
    suggestions = get_treatment_suggestions(
        prediction, is_anemia, subtype,
        hb, mcv, mch, mchc, gender_str, severity["score"]
    )
    return jsonify({"severity": severity, "suggestions": suggestions, "subtype": subtype})


@app.route("/send-report", methods=["POST"])
def send_report():
    """Build the full report HTML and email it via Brevo to any address.

    Brevo free tier: 300 emails/day, no sandbox, send to anyone.

    Request JSON:
        to_email, prediction, is_anemia, confidence, message,
        model_used, params{gender,hemoglobin,mcv,mch,mchc},
        severity, shap, treatment

    Response JSON:
        { success, message, brevo_id, csv_saved, csv_file }
    """
    if not BREVO_API_KEY:
        return jsonify({
            "error": "Brevo is not configured. "
                     "Set BREVO_API_KEY in your .env file and restart."
        }), 503

    body       = request.get_json(force=True) or {}
    to_email   = body.get("to_email",   "").strip()
    prediction = body.get("prediction", "")
    is_anemia  = bool(body.get("is_anemia", False))
    confidence = body.get("confidence", 0)
    message    = body.get("message",    "")
    model_used = body.get("model_used", "RandomForestClassifier")
    params     = body.get("params",     {})
    severity   = body.get("severity",   None)
    shap       = body.get("shap",       None)
    treatment  = body.get("treatment",  None)

    # Validate recipient email
    if not to_email or not re.match(r"[^\s@]+@[^\s@]+\.[^\s@]+", to_email):
        return jsonify({"error": "Please provide a valid email address."}), 400

    html    = build_report_html(prediction, is_anemia, confidence,
                                message, model_used, params,
                                severity=severity, shap=shap, treatment=treatment)
    subject = f"HemoScan AI — {prediction} ({confidence}% confidence)"

    try:
        csv_saved = save_report_as_csv(
            prediction, is_anemia, confidence, message, model_used, params
        )
        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            json={
                "sender":      {"name": BREVO_SENDER_NAME, "email": BREVO_SENDER_EMAIL},
                "to":          [{"email": to_email}],
                "subject":     subject,
                "htmlContent": html,
            },
            headers={
                "accept":       "application/json",
                "content-type": "application/json",
                "api-key":      BREVO_API_KEY,
            },
            timeout=15,
        )

        if resp.status_code not in (200, 201):
            return jsonify(
                {"error": f"Brevo error: {resp.json().get('message', resp.text)}"}
            ), resp.status_code

        return jsonify({
            "success":   True,
            "message":   f"Report sent to {to_email}",
            "brevo_id":  resp.json().get("messageId", ""),
            "csv_saved": csv_saved,
            "csv_file":  REPORT_CSV_PATH if csv_saved else None,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/export-report", methods=["POST"])
def export_report():
    """Build and return the full report HTML for browser print / PDF save.

    No email needed — designed for offline use in rural clinics.
    The frontend opens the HTML in a new tab and calls window.print().
    The user can print to paper or save as PDF via the browser's
    built-in 'Save as PDF' printer.

    Also appends the report to last_report.csv for history tracking.

    Request JSON: prediction, is_anemia, confidence, message,
                  model_used, params, severity, shap, treatment

    Response JSON: { success, html }
    """
    body       = request.get_json(force=True) or {}
    prediction = body.get("prediction", "")
    is_anemia  = bool(body.get("is_anemia", False))
    confidence = body.get("confidence", 0)
    message    = body.get("message",    "")
    model_used = body.get("model_used", "RandomForestClassifier")
    params     = body.get("params",     {})
    severity   = body.get("severity",   None)
    shap       = body.get("shap",       None)
    treatment  = body.get("treatment",  None)

    html = build_report_html(prediction, is_anemia, confidence,
                             message, model_used, params,
                             severity=severity, shap=shap, treatment=treatment)

    # Save to history CSV even for exported (non-emailed) reports
    save_report_as_csv(prediction, is_anemia, confidence,
                       message, model_used, params)

    return jsonify({"success": True, "html": html})


# ══════════════════════════════════════════════════════════════════════
# SECTION 17 — Application Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nHemoScan AI running at http://localhost:5000\n")
    app.run(debug=True)