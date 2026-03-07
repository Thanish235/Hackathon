/* ════════════════════════════════════════════════════════════════════
   HemoScan AI  —  Frontend Application Logic
   ════════════════════════════════════════════════════════════════════
   Sections
     1.  Global State
     2.  Parameter Configuration
     3.  Gender Selector
     4.  Slider ↔ Input Sync
     5.  Real-time Parameter Evaluation
     6.  Form Reset
     7.  Input Validation
     8.  Toast Notifications
     9.  Loading Overlay
    10.  Prediction  (POST /predict)
    11.  Results Rendering
    12.  Risk Severity Gauge  (Feature 3)
    13.  SHAP Explainability Chart  (Feature 8)
    14.  AI Treatment Suggestions  (Feature 1)
    15.  Navigation
    16.  AI File Upload & CBC Auto-Extraction
    17.  Email Modal  (POST /send-report via Brevo)
    18.  Export Report  (POST /export-report → print/PDF)
    19.  Initialisation
   ════════════════════════════════════════════════════════════════════ */

"use strict";

/* ════════════════════════════════════════════════════════════════════
      SECTION 1 — Global State
      Variables persisted across function calls and reused by the email
      modal, export function, and treatment panel.
      ════════════════════════════════════════════════════════════════════ */

/** Currently selected biological sex ('male' | 'female') */
let selectedGender = "male";

/** Last successful prediction result — used by email modal & export */
let lastPredictionData = null;

/** Last CBC parameter values — used by email modal & export */
let lastParamData = null;

/** Last computed risk severity object — included in email & export */
let lastSeverityData = null;

/** Last SHAP importances list — included in email & export */
let lastShapData = null;

/** Last AI treatment suggestions — included in email & export */
let lastTreatmentData = null;

/* ════════════════════════════════════════════════════════════════════
      SECTION 2 — Parameter Configuration
      Single source of truth for each CBC parameter: input/slider IDs,
      value range, normal range, label, and unit.
      ════════════════════════════════════════════════════════════════════ */

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

/* ════════════════════════════════════════════════════════════════════
      SECTION 3 — Gender Selector
      ════════════════════════════════════════════════════════════════════ */

/**
 * Marks the clicked gender toggle as active and updates global state.
 * Called via onclick="selectGender(this)" on each .toggle-btn.
 */
function selectGender(btn) {
  document
    .querySelectorAll(".toggle-btn")
    .forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  selectedGender = btn.dataset.value;
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 4 — Slider ↔ Input Sync
      Keeps the number input and range slider in two-way sync,
      and redraws the custom fill bar on every change.
      ════════════════════════════════════════════════════════════════════ */

/**
 * Called when a number input changes.
 * Mirrors the value onto the corresponding range slider + fill bar.
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

/* ════════════════════════════════════════════════════════════════════
      SECTION 5 — Real-time Parameter Evaluation
      Shows a coloured status line (Low / High / Normal) below each
      input field as the user types or drags the slider.
      ════════════════════════════════════════════════════════════════════ */

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
 * cls is one of: 'low' | 'high' | 'normal'
 */
function getStatusTag(key, value) {
  const cfg = PARAMS[key];
  if (value < cfg.normalMin) return { label: "Low", cls: "low" };
  if (value > cfg.normalMax) return { label: "High", cls: "high" };
  return { label: "Normal", cls: "normal" };
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 6 — Form Reset
      Clears all inputs, sliders, fill bars, status indicators, results
      section, and all global state variables.
      ════════════════════════════════════════════════════════════════════ */

function resetForm() {
  // Clear number inputs
  ["hemoglobin", "mcv", "mch", "mchc"].forEach((id) => {
    document.getElementById(id).value = "";
  });

  // Reset sliders, fill bars, and status labels
  Object.keys(PARAMS).forEach((key) => {
    const cfg = PARAMS[key];
    document.getElementById(cfg.sliderId).value = cfg.min;
    document.getElementById(cfg.fillId).style.width = "0%";
    document.getElementById(cfg.statusId).textContent = "";
    document.getElementById(cfg.statusId).className = "param-status";
  });

  // Hide results section and remove model badge
  document.getElementById("results").style.display = "none";
  const badge = document.getElementById("modelBadge");
  if (badge) badge.remove();

  // Reset gender toggle to Male
  selectedGender = "male";
  document.querySelectorAll(".toggle-btn").forEach((btn, i) => {
    btn.classList.toggle("active", i === 0);
  });

  // Clear all global state
  lastPredictionData = null;
  lastParamData = null;
  lastSeverityData = null;
  lastShapData = null;
  lastTreatmentData = null;
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 7 — Input Validation
      Checks all four CBC fields are filled with valid non-negative numbers
      before allowing a prediction to proceed.
      ════════════════════════════════════════════════════════════════════ */

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

/* ════════════════════════════════════════════════════════════════════
      SECTION 8 — Toast Notifications
      Displays a brief non-blocking notification at the bottom of the
      screen. Auto-dismisses after 3.5 seconds.
      ════════════════════════════════════════════════════════════════════ */

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

  // Inject keyframe animation once
  if (!document.getElementById("toast-style")) {
    const s = document.createElement("style");
    s.id = "toast-style";
    s.textContent =
      "@keyframes toastIn{from{opacity:0;transform:translateX(-50%)" +
      " translateY(10px)}to{opacity:1;transform:translateX(-50%) translateY(0)}}";
    document.head.appendChild(s);
  }

  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 9 — Loading Overlay
      Animates the three loading steps (Validating → Running AI → Generating)
      while the prediction request is in flight.
      ════════════════════════════════════════════════════════════════════ */

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

/* ════════════════════════════════════════════════════════════════════
      SECTION 10 — Prediction  (POST /predict)
   
      Model  : RandomForestClassifier (model5.pkl, 100 trees)
      Input  : Gender (0/1), Hemoglobin, MCH, MCHC, MCV
      Output : prediction, confidence, severity score, SHAP importances
      ════════════════════════════════════════════════════════════════════ */

async function runPrediction() {
  if (!validateInputs()) return;

  const hemoglobin = parseFloat(document.getElementById("hemoglobin").value);
  const mcv = parseFloat(document.getElementById("mcv").value);
  const mch = parseFloat(document.getElementById("mch").value);
  const mchc = parseFloat(document.getElementById("mchc").value);

  // Show loading overlay
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

    // Persist all data for email modal, export, and treatment fetch
    lastPredictionData = data;
    lastParamData = { hemoglobin, mcv, mch, mchc };
    lastSeverityData = data.severity || null;
    lastShapData = data.shap || null;

    displayResults(data, { hemoglobin, mcv, mch, mchc });

    // Fetch treatment suggestions asynchronously — does not block UI
    fetchTreatmentSuggestions(data, { hemoglobin, mcv, mch, mchc });
  } catch (_err) {
    showToast("Cannot reach the server. Make sure app.py is running.");
  }
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 11 — Results Rendering
      Populates the verdict card, confidence bar, and parameter summary
      grid from the prediction response.
      ════════════════════════════════════════════════════════════════════ */

function displayResults(data, params) {
  const isAnemia = data.is_anemia;
  const confidence = data.confidence || 0;

  // ── Verdict card ────────────────────────────────────────────────
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

  // ── Model badge ─────────────────────────────────────────────────
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

  // ── Confidence bar (animated) ────────────────────────────────────
  confVal.textContent = confidence.toFixed(1) + "%";
  confBar.style.width = "0%";
  confBar.style.background = isAnemia ? "#e8472a" : "#2adb7a";
  setTimeout(() => {
    confBar.style.width = confidence + "%";
  }, 100);

  // ── Parameter summary grid ───────────────────────────────────────
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
         <div class="sum-value" style="color:#c07ef0;font-size:1.4rem;padding:.25rem 0">
           ${genderLabel}
         </div>
         <div class="sum-unit">biological sex</div>
         <span class="sum-tag gender">Input</span>
       </div>`;

  grid.innerHTML = paramCards + genderCard;

  // ── Render new feature panels ────────────────────────────────────
  if (data.severity) renderSeverityGauge(data.severity);
  if (data.shap) renderShapChart(data.shap, data.is_anemia);

  // Show and scroll to results
  const resultsEl = document.getElementById("results");
  resultsEl.style.display = "block";
  setTimeout(() => {
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 12 — Risk Severity Gauge  (Feature 3)
      Animates the SVG arc, rotates the needle, counts up the score
      number, and renders per-parameter deviation mini-bars.
      ════════════════════════════════════════════════════════════════════ */

function renderSeverityGauge(severity) {
  const score = severity.score || 0;
  const label = severity.label || "—";
  const level = severity.level || "minimal";
  const perParam = severity.per_param || {};

  const colourMap = {
    minimal: "#2adb7a",
    mild: "#f0b429",
    moderate: "#e8a12a",
    severe: "#e8472a",
  };
  const colour = colourMap[level] || "#e8472a";

  // SVG arc total length ≈ 251.2 for a half-circle with r=80
  const ARC_LEN = 251.2;
  const offset = ARC_LEN - (score / 100) * ARC_LEN;

  const fill = document.getElementById("gaugeFill");
  const needle = document.getElementById("gaugeNeedle");
  const scoreEl = document.getElementById("gaugeScore");
  const labelEl = document.getElementById("gaugeLabel");
  if (!fill) return;

  // Animate arc fill
  fill.style.stroke = colour;
  fill.style.strokeDashoffset = offset;

  // Rotate needle: –90° (score=0) → +90° (score=100)
  const deg = -90 + (score / 100) * 180;
  needle.style.transform = `rotate(${deg}deg)`;

  // Count-up animation for the score number
  let current = 0;
  const step = score / 40;
  const timer = setInterval(() => {
    current = Math.min(current + step, score);
    scoreEl.textContent = Math.round(current);
    if (current >= score) clearInterval(timer);
  }, 25);

  labelEl.textContent = label;
  labelEl.style.color = colour;

  // Per-parameter deviation mini-bars
  const paramNames = {
    Hemoglobin: "Hgb",
    MCV: "MCV",
    MCH: "MCH",
    MCHC: "MCHC",
  };
  const container = document.getElementById("severityParams");

  container.innerHTML = Object.entries(perParam)
    .map(([key, pct]) => {
      const barColor = pct > 60 ? "#e8472a" : pct > 30 ? "#f0b429" : "#2adb7a";
      return `
         <div class="sev-param-row">
           <span class="sev-param-name">${paramNames[key] || key}</span>
           <div class="sev-bar-track">
             <div class="sev-bar-fill"
                  style="width:${pct}%;background:${barColor};"></div>
           </div>
           <span class="sev-param-pct" style="color:${barColor}">${pct}%</span>
         </div>`;
    })
    .join("");
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 13 — SHAP-style Explainability Chart  (Feature 8)
      Renders animated horizontal bars with staggered entry delays.
      Red = value above normal centre | Green = value below normal centre.
      ════════════════════════════════════════════════════════════════════ */

function renderShapChart(shapData, isAnemia) {
  const container = document.getElementById("shapChart");
  if (!container || !shapData || !shapData.length) return;

  const dirLabel = { high: "↑ Above centre", low: "↓ Below centre" };

  container.innerHTML = shapData
    .map((item, i) => {
      const colour = item.direction === "high" ? "#e8472a" : "#2adb7a";
      const delay = i * 80;
      return `
         <div class="shap-row" style="animation-delay:${delay}ms">
           <div class="shap-feat-name">${item.feature}</div>
           <div class="shap-bar-wrap">
             <div class="shap-bar-fill"
                  style="width:0%;background:${colour};
                         transition:width .9s cubic-bezier(.4,0,.2,1) ${delay}ms;"
                  data-target="${item.pct}">
             </div>
           </div>
           <div class="shap-pct" style="color:${colour}">${item.pct}%</div>
           <div class="shap-dir">${dirLabel[item.direction] || ""}</div>
         </div>`;
    })
    .join("");

  // Trigger bar width animations after paint
  requestAnimationFrame(() => {
    container.querySelectorAll(".shap-bar-fill").forEach((el) => {
      el.style.width = el.dataset.target + "%";
    });
  });
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 14 — AI Treatment Suggestions  (Feature 1)
      Fetches from POST /treatment asynchronously after prediction so it
      never blocks the results UI. Stores the result in lastTreatmentData
      for inclusion in email and export reports.
      ════════════════════════════════════════════════════════════════════ */

async function fetchTreatmentSuggestions(predData, paramData) {
  const loadingEl = document.getElementById("treatmentLoading");
  const bodyEl = document.getElementById("treatmentBody");

  loadingEl.style.display = "flex";
  bodyEl.style.display = "none";

  try {
    const resp = await fetch("/treatment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction: predData.prediction,
        is_anemia: predData.is_anemia,
        gender: selectedGender,
        hemoglobin: paramData.hemoglobin,
        mcv: paramData.mcv,
        mch: paramData.mch,
        mchc: paramData.mchc,
      }),
    });

    const data = await resp.json();
    const s = data.suggestions || {};
    if (s.error) throw new Error(s.error);

    // Store globally for email / export
    lastTreatmentData = s;

    // ── Render treatment cards ───────────────────────────────────
    const listSection = (title, emoji, items) => {
      if (!items || !items.length) return "";
      return `
           <div class="treat-section">
             <div class="treat-sec-title">${emoji} ${title}</div>
             <ul class="treat-list">
               ${items.map((i) => `<li>${i}</li>`).join("")}
             </ul>
           </div>`;
    };

    bodyEl.innerHTML = `
         ${s.summary ? `<div class="treat-summary">${s.summary}</div>` : ""}
         <div class="treat-grid">
           ${listSection("Diet", "🥗", s.diet)}
           ${listSection("Supplements", "💊", s.supplements)}
           ${listSection("Lifestyle", "🏃", s.lifestyle)}
         </div>
         ${
           s.when_to_see_doctor
             ? `<div class="treat-doctor"><span>🏥</span>
              <span>${s.when_to_see_doctor}</span></div>`
             : ""
         }
         ${
           s.disclaimer
             ? `<div class="treat-disclaimer">${s.disclaimer}</div>`
             : ""
         }`;

    loadingEl.style.display = "none";
    bodyEl.style.display = "block";
  } catch (err) {
    loadingEl.style.display = "none";
    bodyEl.innerHTML = `<div class="treat-error">⚠ Could not load suggestions: ${err.message}</div>`;
    bodyEl.style.display = "block";
  }
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 15 — Navigation
      ════════════════════════════════════════════════════════════════════ */

/** Hides the results section and scrolls back to the top of the form. */
function newPrediction() {
  document.getElementById("results").style.display = "none";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 16 — AI File Upload & CBC Auto-Extraction
      Handles drag-and-drop and file-picker uploads, then calls
      POST /extract to auto-fill the CBC form fields.
      ════════════════════════════════════════════════════════════════════ */

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
  status.textContent = "⏳  Sending to AI...";
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

    // Map extracted values back to form fields and sync sliders
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
        ? "✓  All 4 values extracted — ready to predict"
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

/* ════════════════════════════════════════════════════════════════════
      SECTION 17 — Email Modal  (POST /send-report via Brevo)
      Opens a modal where the user enters a recipient email address,
      then POSTs the full report data (including severity, SHAP, and
      treatment) to /send-report which emails it via Brevo.
      ════════════════════════════════════════════════════════════════════ */

/** Opens the email modal and pre-fills the prediction preview strip. */
function openEmailModal() {
  if (!lastPredictionData) {
    showToast("Run a prediction first before sending a report.");
    return;
  }

  // Populate preview strip with current prediction
  const prevVerdict = document.getElementById("previewVerdict");
  const prevConf = document.getElementById("previewConfidence");
  prevVerdict.textContent = lastPredictionData.prediction;
  prevVerdict.className =
    "preview-verdict " + (lastPredictionData.is_anemia ? "anemia" : "normal");
  prevConf.textContent = `${lastPredictionData.confidence}% confidence`;

  // Reset modal state
  document.getElementById("emailInput").value = "";
  document.getElementById("modalStatus").textContent = "";
  document.getElementById("modalStatus").className = "modal-status";
  document.getElementById("btnSendEmail").disabled = false;
  document.getElementById("btnSendLabel").textContent = "Send Report";

  document.getElementById("emailModal").classList.add("active");
  setTimeout(() => document.getElementById("emailInput").focus(), 150);
}

/** Closes the email modal. */
function closeEmailModal() {
  document.getElementById("emailModal").classList.remove("active");
}

/** Closes the modal only when clicking the backdrop (not the modal itself). */
function handleModalBackdrop(event) {
  if (event.target === document.getElementById("emailModal")) closeEmailModal();
}

/** Escape key closes the modal. */
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeEmailModal();
});

/**
 * Sends the full prediction report to the entered email address.
 * Includes severity, SHAP, and treatment data if already loaded.
 * Note: treatment data is fetched async — wait for the treatment
 * panel to finish loading before sending for a complete report.
 */
async function sendReportEmail() {
  const emailInput = document.getElementById("emailInput");
  const status = document.getElementById("modalStatus");
  const btn = document.getElementById("btnSendEmail");
  const btnLabel = document.getElementById("btnSendLabel");
  const email = emailInput.value.trim();

  // Client-side email format validation
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    status.textContent = "✗  Please enter a valid email address.";
    status.className = "modal-status error";
    emailInput.focus();
    return;
  }

  btn.disabled = true;
  btnLabel.textContent = "Sending...";
  status.textContent = "⏳  Sending report via Brevo...";
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
        severity: lastSeverityData,
        shap: lastShapData,
        treatment: lastTreatmentData,
      }),
    });

    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error);

    status.textContent = `✓  Report sent to ${email}`;
    status.className = "modal-status success";
    btnLabel.textContent = "Sent ✓";
    setTimeout(() => closeEmailModal(), 2500);
  } catch (err) {
    status.textContent = `✗  ${err.message}`;
    status.className = "modal-status error";
    btn.disabled = false;
    btnLabel.textContent = "Send Report";
  }
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 18 — Export Report  (POST /export-report → print / PDF)
   
      Sends the full report data to POST /export-report which returns the
      complete inline-styled HTML. The HTML is opened in a new browser
      tab and window.print() is triggered automatically.
   
      The user can then:
        • Print to paper on any connected printer
        • Save as PDF using "Save as PDF" in the browser's print dialog
        • Share the file digitally without needing email
   
      This is especially useful in rural areas or clinics without
      reliable email access.
      ════════════════════════════════════════════════════════════════════ */

async function exportReport() {
  if (!lastPredictionData) {
    showToast("Run a prediction first before exporting a report.");
    return;
  }

  const btn = document.getElementById("btnExportReport");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Preparing...";
  }

  try {
    const response = await fetch("/export-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
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
        severity: lastSeverityData,
        shap: lastShapData,
        treatment: lastTreatmentData,
      }),
    });

    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error);

    // Open the returned HTML in a new tab and trigger print dialog
    const win = window.open("", "_blank");
    win.document.write(data.html);
    win.document.close();
    // Small delay ensures the document is fully rendered before printing
    setTimeout(() => win.print(), 600);
  } catch (err) {
    showToast(`Export failed: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Export / Print Report";
    }
  }
}

/* ════════════════════════════════════════════════════════════════════
      SECTION 19 — Initialisation
      Runs once on page load to set the initial fill position of all
      slider track-fill bars from their default values.
      ════════════════════════════════════════════════════════════════════ */

(function initSliderFills() {
  Object.keys(PARAMS).forEach((key) => {
    const cfg = PARAMS[key];
    const slider = document.getElementById(cfg.sliderId);
    const fill = document.getElementById(cfg.fillId);
    const pct = ((slider.value - cfg.min) / (cfg.max - cfg.min)) * 100;
    fill.style.width = pct + "%";
  });
})();
