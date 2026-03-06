/* ============================================================
   HemoScan AI — Application Logic
   File   : main.js
   Backend: Flask + RandomForestClassifier (model5.pkl)
   Gemini : gemini-1.5-flash  (CBC auto-extraction from reports)
   ============================================================ */

"use strict";

/* ════════════════════════════════════════════════════════════
         1.  GLOBAL STATE
         ════════════════════════════════════════════════════════════ */

/** Currently selected biological sex ('male' | 'female' | 'other') */
let selectedGender = "male";

/** Stores the last successful prediction result — used by the email modal */
let lastPredictionData = null;

/** Stores the last CBC param values — used by the email modal */
let lastParamData = null;

/* ════════════════════════════════════════════════════════════
         2.  PARAMETER CONFIGURATION
             Drives input validation, slider behaviour and
             normal-range evaluation for each CBC field.
         ════════════════════════════════════════════════════════════ */
const PARAMS = {
  hgb: {
    inputId: "hemoglobin",
    sliderId: "slider-hgb",
    fillId: "fill-hgb",
    statusId: "status-hgb",
    min: 0,
    max: 25,
    normalMin: 12,
    normalMax: 17,
    label: "Hemoglobin",
    unit: "g/dL",
  },
  mcv: {
    inputId: "mcv",
    sliderId: "slider-mcv",
    fillId: "fill-mcv",
    statusId: "status-mcv",
    min: 50,
    max: 130,
    normalMin: 80,
    normalMax: 100,
    label: "MCV",
    unit: "fL",
  },
  mch: {
    inputId: "mch",
    sliderId: "slider-mch",
    fillId: "fill-mch",
    statusId: "status-mch",
    min: 10,
    max: 50,
    normalMin: 27,
    normalMax: 33,
    label: "MCH",
    unit: "pg",
  },
  mchc: {
    inputId: "mchc",
    sliderId: "slider-mchc",
    fillId: "fill-mchc",
    statusId: "status-mchc",
    min: 20,
    max: 45,
    normalMin: 32,
    normalMax: 36,
    label: "MCHC",
    unit: "g/dL",
  },
};

/* ════════════════════════════════════════════════════════════
         3.  GENDER SELECTOR
         ════════════════════════════════════════════════════════════ */

/**
 * Marks the clicked gender toggle as active and updates state.
 * Called via onclick="selectGender(this)" on each .toggle-btn.
 * @param {HTMLButtonElement} btn
 */
function selectGender(btn) {
  document
    .querySelectorAll(".toggle-btn")
    .forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  selectedGender = btn.dataset.value;
}

/* ════════════════════════════════════════════════════════════
         4.  SLIDER ↔ INPUT SYNC
         ════════════════════════════════════════════════════════════ */

/**
 * Called when a number input changes.
 * Mirrors the value onto the range slider and redraws the fill bar.
 */
function updateSlider(key, value, min, max) {
  const cfg = PARAMS[key];
  const slider = document.getElementById(cfg.sliderId);
  const fill = document.getElementById(cfg.fillId);

  if (value !== "") {
    slider.value = value;
    const pct = ((value - min) / (max - min)) * 100;
    fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
  }

  evaluateParam(key, parseFloat(value));
}

/**
 * Called when a range slider changes.
 * Mirrors the value back into the corresponding number input.
 */
function syncInput(inputId, value) {
  const input = document.getElementById(inputId);
  input.value = value;

  const key = Object.keys(PARAMS).find((k) => PARAMS[k].inputId === inputId);
  if (!key) return;

  const cfg = PARAMS[key];
  const pct = ((value - cfg.min) / (cfg.max - cfg.min)) * 100;
  document.getElementById(cfg.fillId).style.width = pct + "%";
  evaluateParam(key, parseFloat(value));
}

/* ════════════════════════════════════════════════════════════
         5.  REAL-TIME PARAMETER EVALUATION
         ════════════════════════════════════════════════════════════ */

/**
 * Shows a coloured status line below each parameter field
 * (low / high / normal) as the user types or drags.
 */
function evaluateParam(key, value) {
  const cfg = PARAMS[key];
  const statusEl = document.getElementById(cfg.statusId);

  if (isNaN(value) || value === "") {
    statusEl.textContent = "";
    statusEl.className = "param-status";
    return;
  }

  if (value < cfg.normalMin) {
    statusEl.textContent = `↓ Below normal (< ${cfg.normalMin})`;
    statusEl.className = "param-status low";
  } else if (value > cfg.normalMax) {
    statusEl.textContent = `↑ Above normal (> ${cfg.normalMax})`;
    statusEl.className = "param-status high";
  } else {
    statusEl.textContent = "✓ Within normal range";
    statusEl.className = "param-status normal";
  }
}

/**
 * Returns { label, cls } for use in the results summary cards.
 */
function getStatusTag(key, value) {
  const cfg = PARAMS[key];
  if (value < cfg.normalMin) return { label: "Low", cls: "low" };
  if (value > cfg.normalMax) return { label: "High", cls: "high" };
  return { label: "Normal", cls: "normal" };
}

/* ════════════════════════════════════════════════════════════
         6.  FORM RESET
         ════════════════════════════════════════════════════════════ */

function resetForm() {
  ["hemoglobin", "mcv", "mch", "mchc"].forEach((id) => {
    document.getElementById(id).value = "";
  });

  Object.keys(PARAMS).forEach((key) => {
    const cfg = PARAMS[key];
    document.getElementById(cfg.sliderId).value = cfg.min;
    document.getElementById(cfg.fillId).style.width = "0%";
    document.getElementById(cfg.statusId).textContent = "";
    document.getElementById(cfg.statusId).className = "param-status";
  });

  document.getElementById("results").style.display = "none";

  const badge = document.getElementById("modelBadge");
  if (badge) badge.remove();

  selectedGender = "male";
  lastPredictionData = null;
  lastParamData = null;
  document.querySelectorAll(".toggle-btn").forEach((btn, i) => {
    btn.classList.toggle("active", i === 0);
  });
}

/* ════════════════════════════════════════════════════════════
         7.  INPUT VALIDATION
         ════════════════════════════════════════════════════════════ */

function validateInputs() {
  const fields = [
    { id: "hemoglobin", name: "Hemoglobin" },
    { id: "mcv", name: "MCV" },
    { id: "mch", name: "MCH" },
    { id: "mchc", name: "MCHC" },
  ];

  for (const field of fields) {
    const raw = document.getElementById(field.id).value;
    const val = parseFloat(raw);

    if (raw === "" || isNaN(val)) {
      showToast(`Please enter a valid value for ${field.name}`);
      document.getElementById(field.id).focus();
      return false;
    }

    if (val < 0) {
      showToast(`${field.name} cannot be negative`);
      return false;
    }
  }

  return true;
}

/* ════════════════════════════════════════════════════════════
         8.  TOAST NOTIFICATIONS
         ════════════════════════════════════════════════════════════ */

function showToast(message) {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  toast.style.cssText = [
    "position:fixed",
    "bottom:2rem",
    "left:50%",
    "transform:translateX(-50%)",
    "background:#1e2126",
    "border:1px solid rgba(232,71,42,.4)",
    "border-radius:8px",
    "padding:.9rem 1.5rem",
    "color:#e8472a",
    'font-family:"IBM Plex Mono",monospace',
    "font-size:.85rem",
    "z-index:9999",
    "box-shadow:0 4px 20px rgba(0,0,0,.5)",
    "animation:toastIn .3s ease",
  ].join(";");

  if (!document.getElementById("toast-style")) {
    const s = document.createElement("style");
    s.id = "toast-style";
    s.textContent =
      "@keyframes toastIn{from{opacity:0;transform:translateX(-50%) translateY(10px)}to{opacity:1;transform:translateX(-50%) translateY(0)}}";
    document.head.appendChild(s);
  }

  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

/* ════════════════════════════════════════════════════════════
         9.  LOADING OVERLAY STEPS
         ════════════════════════════════════════════════════════════ */

function animateLoadingSteps() {
  const stepIds = ["step1", "step2", "step3"];
  let current = 0;

  return new Promise((resolve) => {
    function advance() {
      if (current > 0) {
        const prev = document.getElementById(stepIds[current - 1]);
        prev.classList.remove("active");
        prev.classList.add("done");
      }
      if (current < stepIds.length) {
        document.getElementById(stepIds[current]).classList.add("active");
        current++;
        setTimeout(advance, 700);
      } else {
        setTimeout(resolve, 400);
      }
    }
    advance();
  });
}

function resetLoadingSteps() {
  ["step1", "step2", "step3"].forEach((id) => {
    const el = document.getElementById(id);
    el.classList.remove("active", "done");
  });
}

/* ════════════════════════════════════════════════════════════
         10. PREDICTION  (RandomForestClassifier via Flask /predict)
      
         Model  : model5.pkl — RandomForestClassifier (100 trees)
         Input  : ['Gender','Hemoglobin','MCH','MCHC','MCV']
         Gender : 0=Female | 1=Male/Other  (encoded by app.py)
         Output : 0=No Anemia | 1=Anemia  + predict_proba confidence
         ════════════════════════════════════════════════════════════ */

async function runPrediction() {
  if (!validateInputs()) return;

  const hemoglobin = parseFloat(document.getElementById("hemoglobin").value);
  const mcv = parseFloat(document.getElementById("mcv").value);
  const mch = parseFloat(document.getElementById("mch").value);
  const mchc = parseFloat(document.getElementById("mchc").value);

  const overlay = document.getElementById("loadingOverlay");
  overlay.classList.add("active");
  resetLoadingSteps();
  await animateLoadingSteps();
  overlay.classList.remove("active");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hemoglobin,
        mcv,
        mch,
        mchc,
        gender: selectedGender,
      }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showToast(data.error || "Prediction failed — check the Flask server.");
      return;
    }

    // Save for email modal use
    lastPredictionData = data;
    lastParamData = { hemoglobin, mcv, mch, mchc };

    displayResults(data, { hemoglobin, mcv, mch, mchc });
  } catch (_err) {
    showToast("Cannot reach the server. Make sure app.py is running.");
  }
}

/* ════════════════════════════════════════════════════════════
         11. RESULTS RENDERING
         ════════════════════════════════════════════════════════════ */

function displayResults(data, params) {
  const isAnemia = data.is_anemia;
  const confidence = data.confidence || 0;

  const card = document.getElementById("verdictCard");
  const iconWrap = document.getElementById("verdictIconWrap");
  const icon = document.getElementById("verdictIcon");
  const title = document.getElementById("verdictTitle");
  const subtitle = document.getElementById("verdictSubtitle");
  const confVal = document.getElementById("confValue");
  const confBar = document.getElementById("confBar");

  card.className = "verdict-card " + (isAnemia ? "anemia" : "normal");
  iconWrap.className = "verdict-icon-wrap " + (isAnemia ? "anemia" : "normal");
  icon.textContent = isAnemia ? "🩸" : "✅";
  title.className = "verdict-title " + (isAnemia ? "anemia" : "normal");
  title.textContent = data.prediction;
  subtitle.textContent = data.message || "";

  // ── Model badge ──
  const modelUsed = data.model_used || "RandomForestClassifier";
  let badge = document.getElementById("modelBadge");
  if (!badge) {
    badge = document.createElement("div");
    badge.id = "modelBadge";
    badge.style.cssText = [
      "margin-top:.75rem",
      "display:inline-flex",
      "align-items:center",
      "gap:.4rem",
      'font-family:"IBM Plex Mono",monospace',
      "font-size:.7rem",
      "letter-spacing:.04em",
      "color:#5c5751",
      "background:rgba(255,255,255,.04)",
      "border:1px solid rgba(255,255,255,.09)",
      "padding:.25rem .7rem",
      "border-radius:4px",
    ].join(";");
    document.querySelector(".verdict-content").appendChild(badge);
  }
  badge.innerHTML = `🌲&nbsp;${modelUsed}`;

  // ── Confidence bar ──
  confVal.textContent = confidence.toFixed(1) + "%";
  confBar.style.width = "0%";
  confBar.style.background = isAnemia ? "#e8472a" : "#2adb7a";
  setTimeout(() => {
    confBar.style.width = confidence + "%";
  }, 100);

  // ── Summary cards ──
  const grid = document.getElementById("summaryGrid");

  const valueColor = (cls) =>
    cls === "low"
      ? "var(--blood)"
      : cls === "high"
      ? "var(--yellow)"
      : "var(--green)";

  const paramList = [
    { key: "hgb", val: params.hemoglobin, label: "Hemoglobin", unit: "g/dL" },
    { key: "mcv", val: params.mcv, label: "MCV", unit: "fL" },
    { key: "mch", val: params.mch, label: "MCH", unit: "pg" },
    { key: "mchc", val: params.mchc, label: "MCHC", unit: "g/dL" },
  ];

  const paramCards = paramList
    .map((p) => {
      const tag = getStatusTag(p.key, p.val);
      const cardCls = tag.cls !== "normal" ? "abnormal" : "normal-item";
      return `
            <div class="summary-item ${cardCls}">
              <div class="sum-param">${p.label}</div>
              <div class="sum-value" style="color:${valueColor(tag.cls)}">${
        p.val
      }</div>
              <div class="sum-unit">${p.unit}</div>
              <span class="sum-tag ${tag.cls}">${tag.label}</span>
            </div>`;
    })
    .join("");

  const genderLabel =
    selectedGender.charAt(0).toUpperCase() + selectedGender.slice(1);
  const genderCard = `
          <div class="summary-item gender-item">
            <div class="sum-param">Sex</div>
            <div class="sum-value" style="color:#c07ef0;font-size:1.4rem;padding:.25rem 0">${genderLabel}</div>
            <div class="sum-unit">biological sex</div>
            <span class="sum-tag gender">Input</span>
          </div>`;

  grid.innerHTML = paramCards + genderCard;

  const resultsEl = document.getElementById("results");
  resultsEl.style.display = "block";
  setTimeout(() => {
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

/* ════════════════════════════════════════════════════════════
         12. NAVIGATION
         ════════════════════════════════════════════════════════════ */

function newPrediction() {
  document.getElementById("results").style.display = "none";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ════════════════════════════════════════════════════════════
         13. EXPORT REPORT
         ════════════════════════════════════════════════════════════ */

function printReport() {
  const titleEl = document.getElementById("verdictTitle");
  const subtitleEl = document.getElementById("verdictSubtitle");
  const confEl = document.getElementById("confValue");
  const badgeEl = document.getElementById("modelBadge");
  const modelUsed = badgeEl
    ? badgeEl.textContent.replace("🌲", "").trim()
    : "RandomForestClassifier";

  const paramList = [
    { label: "Hemoglobin", id: "hemoglobin", unit: "g/dL" },
    { label: "MCV", id: "mcv", unit: "fL" },
    { label: "MCH", id: "mch", unit: "pg" },
    { label: "MCHC", id: "mchc", unit: "g/dL" },
  ];

  const tableRows =
    `<tr><td>Biological Sex</td><td>${selectedGender}</td><td>—</td></tr>` +
    paramList
      .map((p) => {
        const val = document.getElementById(p.id).value;
        const key = Object.keys(PARAMS).find((k) => PARAMS[k].inputId === p.id);
        const tag = key ? getStatusTag(key, parseFloat(val)) : { label: "—" };
        return `<tr><td>${p.label}</td><td>${val} ${p.unit}</td><td>${tag.label}</td></tr>`;
      })
      .join("");

  const verdictCls = titleEl.classList.contains("anemia") ? "anemia" : "normal";

  const html = `<!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8"/>
        <title>HemoScan AI — Prediction Report</title>
        <style>
          body{font-family:"Segoe UI",sans-serif;padding:40px;color:#111;max-width:700px;margin:0 auto}
          h1{color:#e8472a;font-size:1.7rem;margin-bottom:.2rem}
          .sub{color:#666;font-size:.9rem;margin-bottom:2rem}
          table{width:100%;border-collapse:collapse;margin:1.5rem 0}
          th,td{padding:.65rem 1rem;border:1px solid #ddd;text-align:left;font-size:.9rem}
          th{background:#f8f8f8;font-weight:600}
          .verdict{padding:1rem 1.25rem;border-radius:8px;margin:1.5rem 0}
          .verdict.anemia{background:#fff2f0;border-left:4px solid #e8472a}
          .verdict.normal{background:#f0fff6;border-left:4px solid #2adb7a}
          .model-tag{display:inline-block;font-family:monospace;font-size:.75rem;
                     background:#f4f4f4;border:1px solid #ddd;padding:.2rem .5rem;
                     border-radius:4px;margin-top:.6rem;color:#555}
          .note{font-size:.78rem;color:#888;margin-top:2rem;border-top:1px solid #eee;padding-top:1rem}
        </style>
      </head>
      <body>
        <h1>HemoScan AI</h1>
        <div class="sub">Anemia Prediction Report</div>
        <table>
          <tr><th>Parameter</th><th>Value</th><th>Status</th></tr>
          ${tableRows}
        </table>
        <div class="verdict ${verdictCls}">
          <strong>${titleEl.textContent}</strong> &mdash; Confidence: ${
    confEl.textContent
  }<br/>
          <span style="font-size:.9rem;color:#555">${
            subtitleEl.textContent
          }</span><br/>
          <span class="model-tag">🌲 ${modelUsed}</span>
        </div>
        <p class="note">
          Generated by ${modelUsed} for clinical decision support only.<br/>
          Not a substitute for professional medical advice.<br/>
          Generated: ${new Date().toLocaleString()}
        </p>
      </body>
      </html>`;

  const win = window.open("", "_blank");
  win.document.write(html);
  win.document.close();
  setTimeout(() => win.print(), 400);
}

/* ════════════════════════════════════════════════════════════
         14. GEMINI FILE UPLOAD & AUTO-EXTRACTION
         ════════════════════════════════════════════════════════════ */

let selectedFile = null;

function handleFileSelect(input) {
  if (input.files && input.files[0]) setUploadFile(input.files[0]);
}

function handleDragOver(event) {
  event.preventDefault();
  document.getElementById("uploadZone").classList.add("drag-over");
}

function handleDrop(event) {
  event.preventDefault();
  document.getElementById("uploadZone").classList.remove("drag-over");
  const file = event.dataTransfer.files[0];
  if (file) setUploadFile(file);
}

function setUploadFile(file) {
  selectedFile = file;
  document.getElementById("uploadZone").classList.add("has-file");
  document.getElementById("uploadFilename").textContent = `✓  ${file.name}  (${(
    file.size / 1024
  ).toFixed(1)} KB)`;
  document.getElementById("btnExtract").disabled = false;
  document.getElementById("extractStatus").textContent = "";
  document.getElementById("extractStatus").className = "extract-status";
}

async function extractValues() {
  if (!selectedFile) return;

  const status = document.getElementById("extractStatus");
  const btn = document.getElementById("btnExtract");

  btn.disabled = true;
  status.textContent = "⏳  Sending to Gemini AI...";
  status.className = "extract-status loading";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/extract", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok || data.error)
      throw new Error(data.error || "Extraction failed");

    const fieldMap = {
      hemoglobin: { key: "hgb", min: 0, max: 25 },
      mcv: { key: "mcv", min: 50, max: 130 },
      mch: { key: "mch", min: 10, max: 50 },
      mchc: { key: "mchc", min: 20, max: 45 },
    };

    let filled = 0;
    for (const [field, cfg] of Object.entries(fieldMap)) {
      const value = data[field];
      if (value !== null && value !== undefined) {
        const input = document.getElementById(field);
        input.value = parseFloat(value).toFixed(1);
        updateSlider(cfg.key, input.value, cfg.min, cfg.max);
        filled++;
      }
    }

    status.textContent =
      filled === 4
        ? `✓  All 4 values extracted — ready to predict`
        : `✓  Extracted ${filled}/4 values — fill in the rest manually`;
    status.className = "extract-status success";

    document.getElementById("predict").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    status.textContent = `✗  ${err.message}`;
    status.className = "extract-status error";
  } finally {
    btn.disabled = false;
  }
}
/* ════════════════════════════════════════════════════════════
         14. EMAIL MODAL  (Resend via Flask /send-report)
         ════════════════════════════════════════════════════════════ */

/**
 * Opens the email modal and pre-fills the preview strip
 * with the latest prediction result.
 */
function openEmailModal() {
  if (!lastPredictionData) {
    showToast("Run a prediction first before sending a report.");
    return;
  }

  // Populate preview strip
  const prevVerdict = document.getElementById("previewVerdict");
  const prevConf = document.getElementById("previewConfidence");
  prevVerdict.textContent = lastPredictionData.prediction;
  prevVerdict.className =
    "preview-verdict " + (lastPredictionData.is_anemia ? "anemia" : "normal");
  prevConf.textContent = `${lastPredictionData.confidence}% confidence`;

  // Reset input + status
  document.getElementById("emailInput").value = "";
  document.getElementById("modalStatus").textContent = "";
  document.getElementById("modalStatus").className = "modal-status";
  document.getElementById("btnSendEmail").disabled = false;
  document.getElementById("btnSendLabel").textContent = "Send Report";

  document.getElementById("emailModal").classList.add("active");
  setTimeout(() => document.getElementById("emailInput").focus(), 150);
}

/** Closes the modal. */
function closeEmailModal() {
  document.getElementById("emailModal").classList.remove("active");
}

/** Closes only when clicking the backdrop (not the modal card itself). */
function handleModalBackdrop(event) {
  if (event.target === document.getElementById("emailModal")) closeEmailModal();
}

/** Escape key closes the modal. */
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeEmailModal();
});

/**
 * Sends the full prediction report to the entered email address
 * by POSTing to Flask /send-report, which uses the Resend API.
 */
async function sendReportEmail() {
  const emailInput = document.getElementById("emailInput");
  const status = document.getElementById("modalStatus");
  const btn = document.getElementById("btnSendEmail");
  const btnLabel = document.getElementById("btnSendLabel");
  const email = emailInput.value.trim();

  // Client-side email format check
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    status.textContent = "✗  Please enter a valid email address.";
    status.className = "modal-status error";
    emailInput.focus();
    return;
  }

  // Loading state
  btn.disabled = true;
  btnLabel.textContent = "Sending...";
  status.textContent = "⏳  Sending report via Resend...";
  status.className = "modal-status loading";

  try {
    const response = await fetch("/send-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        to_email: email,
        prediction: lastPredictionData.prediction,
        is_anemia: lastPredictionData.is_anemia,
        confidence: lastPredictionData.confidence,
        message: lastPredictionData.message,
        model_used: lastPredictionData.model_used || "RandomForestClassifier",
        params: {
          gender: selectedGender,
          hemoglobin: lastParamData.hemoglobin,
          mcv: lastParamData.mcv,
          mch: lastParamData.mch,
          mchc: lastParamData.mchc,
        },
      }),
    });

    const data = await response.json();

    if (!response.ok || data.error) throw new Error(data.error);

    // Success
    status.textContent = `✓  Report sent to ${email}`;
    status.className = "modal-status success";
    btnLabel.textContent = "Sent ✓";

    // Auto-close after 2.5s
    setTimeout(() => closeEmailModal(), 2500);
  } catch (err) {
    status.textContent = `✗  ${err.message}`;
    status.className = "modal-status error";
    btn.disabled = false;
    btnLabel.textContent = "Send Report";
  }
}

/* ════════════════════════════════════════════════════════════
         15. INITIALISATION
         ════════════════════════════════════════════════════════════ */

(function initSliderFills() {
  Object.keys(PARAMS).forEach((key) => {
    const cfg = PARAMS[key];
    const slider = document.getElementById(cfg.sliderId);
    const fill = document.getElementById(cfg.fillId);
    const pct = ((slider.value - cfg.min) / (cfg.max - cfg.min)) * 100;
    fill.style.width = pct + "%";
  });
})();
