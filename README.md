# 🩸 HemoScan AI — Anemia Prediction & Clinical Decision Support System

> AI-powered anemia detection from CBC blood parameters — built at the AVNIET × NASSCOM × SMARTBRIDGE 4-Day Generative AI Hackathon.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat-square&logo=scikit-learn)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

Anemia affects **1.6 billion people worldwide**, yet most cases go undetected because CBC (Complete Blood Count) test results go uninterpreted — especially in rural clinics and primary health centres where no specialist is available.

**HemoScan AI** bridges that gap. Enter just 4 CBC values (or upload your lab report), and get an instant AI-powered prediction, severity score, explainability chart, and a personalised treatment plan — in under 10 seconds.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **Anemia Prediction** | RandomForest model trained on 65,796 balanced patient records — 99.97% accuracy |
| 📄 **Auto-Extract from Report** | Upload a PDF or image → AI extracts CBC values via OCR + Groq LLaMA 3 |
| 📊 **Risk Severity Score** | Weighted 0–100 composite score mapped to Minimal / Mild / Moderate / Severe |
| 🧠 **Prediction Explainability** | SHAP-style feature attribution — see exactly which parameter drove the result |
| 💊 **AI Treatment Suggestions** | Personalised diet, supplements, lifestyle plan & doctor urgency via Groq LLaMA 3 |
| 📧 **Email Report** | Send a full formatted clinical report to any email via Brevo API |
| 🖨️ **Export / Print Report** | Generate a print-ready PDF — works fully offline |
| 📁 **CSV History Log** | Every prediction is appended to `last_report.csv` for audit purposes |

---

## 🛠️ Tech Stack

- **Backend** — Python, Flask
- **ML Model** — scikit-learn (RandomForestClassifier, 100 estimators)
- **Generative AI** — Groq LLaMA 3.1-8b-instant
- **OCR & Parsing** — Tesseract OCR, pdfplumber
- **Email** — Brevo Transactional API
- **Frontend** — Vanilla JS, CSS3 (animations, SVG gauge)
- **Data** — 65,796 balanced CBC records (`BALANCED_ANEMIA_DATASET.csv`)

---

## 📂 Project Structure

```
HemoScan-AI/
├── app.py                      # Flask backend (17 sections)
├── script.js                   # Frontend logic (19 sections)
├── index.html                  # UI layout
├── styles.css                  # Styling & animations
├── model5-2.pkl                # Trained RandomForest model
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── last_report.csv             # Auto-generated prediction log
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Thanish235/Hackathon.git
cd Hackathon
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Tesseract OCR must also be installed on your system.
> - Windows: [Download here](https://github.com/tesseract-ocr/tesseract)
> - Linux: `sudo apt install tesseract-ocr`
> - Mac: `brew install tesseract`

### 3. Configure environment variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Then fill in your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
BREVO_API_KEY=your_brevo_api_key_here
BREVO_SENDER_EMAIL=your_sender_email_here
BREVO_SENDER_NAME=HemoScan AI
```

- Get a free Groq API key at [console.groq.com](https://console.groq.com)
- Get a free Brevo API key at [brevo.com](https://brevo.com)

### 4. Run the app

```bash
python app.py
```

Open your browser at `http://localhost:5000`

---

## 🧪 How It Works

```
User enters CBC values  →  RandomForest model predicts anemia
         ↓
Severity score computed (weighted: Hgb 40%, MCV 25%, MCH 20%, MCHC 15%)
         ↓
SHAP-style feature attribution generated
         ↓
Groq LLaMA 3 generates personalised treatment suggestions
         ↓
Full clinical report → Email (Brevo) or Print/PDF (offline)
```

### CBC Parameters Used

| Parameter | Unit | Normal Range |
|---|---|---|
| Hemoglobin (Hgb) | g/dL | 12–17 (gender-adjusted) |
| Mean Corpuscular Volume (MCV) | fL | 80–100 |
| Mean Corpuscular Hemoglobin (MCH) | pg | 27–33 |
| MCH Concentration (MCHC) | g/dL | 32–36 |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Dataset size | 65,796 samples |
| Class balance | 50% anemia / 50% normal |
| Accuracy | **99.97%** |
| Model | RandomForestClassifier (100 estimators) |
| Feature importance | Hemoglobin: 76.7%, Gender: 8.2%, MCH: 6.6%, MCHC: 5.4%, MCV: 3.1% |

---

## ⚠️ Disclaimer

HemoScan AI is a **clinical decision support tool** designed to assist healthcare workers — it is **not a substitute for professional medical diagnosis or advice**. Always consult a qualified healthcare provider before making any clinical decisions.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
