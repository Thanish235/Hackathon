/* ════════════════════════════════════════════════════════════
   GEMINI FILE UPLOAD & EXTRACTION
   ════════════════════════════════════════════════════════════ */

/** Holds the currently selected File object */
let selectedFile = null;

/** Called when the user picks a file via the file input */
function handleFileSelect(input) {
  if (input.files && input.files[0]) {
    setUploadFile(input.files[0]);
  }
}

/** Prevents default browser behaviour when dragging a file over the zone */
function handleDragOver(event) {
  event.preventDefault();
  document.getElementById("uploadZone").classList.add("drag-over");
}

/** Handles a file being dropped onto the upload zone */
function handleDrop(event) {
  event.preventDefault();
  const zone = document.getElementById("uploadZone");
  zone.classList.remove("drag-over");
  const file = event.dataTransfer.files[0];
  if (file) setUploadFile(file);
}

/**
 * Stores the selected file, updates the UI, and enables the
 * Extract button.
 * @param {File} file
 */
function setUploadFile(file) {
  selectedFile = file;

  const zone = document.getElementById("uploadZone");
  const filename = document.getElementById("uploadFilename");
  const btn = document.getElementById("btnExtract");
  const status = document.getElementById("extractStatus");

  zone.classList.add("has-file");
  filename.textContent = `✓ ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  btn.disabled = false;
  status.textContent = "";
  status.className = "extract-status";
}

/**
 * Sends the uploaded file to the Flask /extract endpoint,
 * waits for Gemini to parse the CBC report, then auto-fills
 * the matching input fields.
 */
async function extractValues() {
  if (!selectedFile) return;

  const status = document.getElementById("extractStatus");
  const btn = document.getElementById("btnExtract");

  // Show loading state
  btn.disabled = true;
  status.textContent = "⏳ Analyzing report with Gemini AI...";
  status.className = "extract-status loading";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/extract", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || "Extraction failed");
    }

    // ── Auto-fill each field that Gemini returned ──
    const fieldMap = {
      hemoglobin: { key: "hgb", min: 0, max: 25 },
      mcv: { key: "mcv", min: 50, max: 130 },
      mch: { key: "mch", min: 10, max: 50 },
      mchc: { key: "mchc", min: 20, max: 45 },
    };

    let filledCount = 0;

    for (const [field, cfg] of Object.entries(fieldMap)) {
      const value = data[field];
      if (value !== null && value !== undefined) {
        const input = document.getElementById(field);
        input.value = parseFloat(value).toFixed(1);
        updateSlider(cfg.key, input.value, cfg.min, cfg.max);
        filledCount++;
      }
    }

    // Success message
    status.textContent =
      filledCount === 4
        ? `✓ All 4 values extracted successfully`
        : `✓ Extracted ${filledCount}/4 values — please fill in the rest manually`;
    status.className = "extract-status success";

    // Scroll down to the CBC form
    document.getElementById("predict").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    status.textContent = `✗ ${err.message}`;
    status.className = "extract-status error";
  } finally {
    btn.disabled = false;
  }
}
