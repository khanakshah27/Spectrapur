window.addEventListener("error", function (e) {
  console.error("Spectrapur: uncaught error ->", e.message, e.filename + ":" + e.lineno);
});

(function () {
  "use strict";

  const CFG = window.SPECTRAPUR_CONFIG || {};
  const API_BASE = CFG.API_BASE_URL || "https://spectrapur-api.onrender.com";
  const MAX_MB = CFG.MAX_FILE_SIZE_MB || 10;
  if (!window.SPECTRAPUR_CONFIG) {
    console.warn("Spectrapur: js/config.js did not load — using fallback API_BASE_URL:", API_BASE);
  }

  // ---- DOM references ----
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const fileChipHolder = document.getElementById("file-chip-holder");
  const pasteInput = document.getElementById("paste-input");
  const sampleToggle = document.getElementById("sample-toggle");
  const runBtn = document.getElementById("run-btn");
  const statusLine = document.getElementById("status-line");
  const heroSampleBtn = document.getElementById("hero-sample-btn");

  const resultsSection = document.getElementById("results");
  const statGrid = document.getElementById("stat-grid");
  const peakTableBody = document.getElementById("peak-table-body");
  const downloadCsvBtn = document.getElementById("download-csv");
  const downloadReportBtn = document.getElementById("download-report");
  const downloadPngBtn = document.getElementById("download-png");
  const toast = document.getElementById("toast");
    // Fail loudly instead of silently: if any required element is missing
  // (stale/partially-deployed HTML, a typo, etc.), show a visible banner
  // and stop here so we never throw mid-way through wiring up listeners.
  const required = {
    dropzone, fileInput, fileChipHolder, pasteInput, sampleToggle, runBtn,
    statusLine, heroSampleBtn, resultsSection, statGrid, peakTableBody,
    downloadCsvBtn, downloadReportBtn, downloadPngBtn, toast,
  };
  const missing = Object.keys(required).filter((k) => !required[k]);
  if (missing.length > 0) {
    console.error("Spectrapur: missing expected page elements:", missing);
    const banner = document.createElement("div");
    banner.style.cssText =
      "position:fixed;top:0;left:0;right:0;z-index:9999;background:#b3663f;color:#fff;" +
      "font-family:sans-serif;font-size:13px;padding:10px 16px;text-align:center;";
    banner.textContent =
      "Spectrapur: page did not load correctly (missing: " + missing.join(", ") +
      "). Hard-refresh the page (Ctrl/Cmd+Shift+R). If it persists, redeploy index.html and js/app.js together — they must match.";
    document.body.prepend(banner);
    return; // stop here; nothing below is safe to run
  }

  let selectedFile = null;
  let lastResult = null;
  let overlayChart = null;
  let residualChart = null;

  // ---- Helpers ----
  function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2600);
  }

  function setStatus(message, kind) {
    statusLine.textContent = message || "";
    statusLine.className = "status-line" + (kind ? " " + kind : "");
  }

  function setRunning(isRunning) {
    runBtn.disabled = isRunning;
    runBtn.innerHTML = isRunning
      ? '<span class="spinner"></span> Denoising…'
      : "Run denoising";
  }

  function fmt(n, digits = 4) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    if (Math.abs(n) >= 1000 || (Math.abs(n) < 0.001 && n !== 0)) {
      return Number(n).toExponential(2);
    }
    return Number(n).toFixed(digits);
  }

  // ---- File selection ----
  function setSelectedFile(file) {
    if (!file) return;
    const sizeMb = file.size / (1024 * 1024);
    if (sizeMb > MAX_MB) {
      setStatus(`"${file.name}" is ${sizeMb.toFixed(1)} MB — the limit is ${MAX_MB} MB.`, "error");
      return;
    }
    selectedFile = file;
    pasteInput.value = "";
    sampleToggle.checked = false;
    fileChipHolder.innerHTML = `<span class="file-chip">📄 ${file.name} — ${(sizeMb).toFixed(2)} MB</span>`;
    setStatus("File ready. Click \u201cRun denoising\u201d.", "");
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) setSelectedFile(e.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) setSelectedFile(file);
  });

  pasteInput.addEventListener("input", () => {
    if (pasteInput.value.trim()) {
      selectedFile = null;
      sampleToggle.checked = false;
      fileChipHolder.innerHTML = "";
    }
  });

  sampleToggle.addEventListener("change", () => {
    if (sampleToggle.checked) {
      selectedFile = null;
      pasteInput.value = "";
      fileChipHolder.innerHTML = "";
      setStatus("Synthetic demo spectrum selected.", "");
    }
  });

  heroSampleBtn.addEventListener("click", () => {
    document.getElementById("workspace").scrollIntoView({ behavior: "smooth" });
    sampleToggle.checked = true;
    selectedFile = null;
    pasteInput.value = "";
    fileChipHolder.innerHTML = "";
    setStatus("Synthetic demo spectrum selected. Click \u201cRun denoising\u201d.", "");
  });

  // ---- Parse pasted text client-side into {x, y} for JSON submission ----
  function parsePastedText(text) {
    const lines = text
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
    if (lines.length === 0) throw new Error("Paste box is empty.");

    const splitLine = (line) => {
      if (line.includes(",")) return line.split(",").map((s) => s.trim()).filter(Boolean);
      if (line.includes("\t")) return line.split("\t").map((s) => s.trim()).filter(Boolean);
      return line.split(/\s+/).filter(Boolean);
    };

    let rows = lines.map(splitLine);
    const isNum = (s) => s !== "" && !Number.isNaN(Number(s));
    if (rows.length && !isNum(rows[0][0])) rows = rows.slice(1); // drop header

    const xs = [];
    const ys = [];
    if (rows[0] && rows[0].length >= 2) {
      rows.forEach((r) => {
        if (r.length >= 2 && isNum(r[0]) && isNum(r[1])) {
          xs.push(Number(r[0]));
          ys.push(Number(r[1]));
        }
      });
    } else {
      rows.forEach((r) => {
        if (r.length >= 1 && isNum(r[0])) ys.push(Number(r[0]));
      });
      for (let i = 0; i < ys.length; i++) xs.push(i);
    }

    if (ys.length < 8) throw new Error("Need at least 8 valid numeric data points.");
    return { x: xs, y: ys };
  }

  // ---- API calls ----
  async function runDenoise() {
    setRunning(true);
    setStatus("Uploading and running the model…", "");

    try {
      let response;

      if (sampleToggle.checked) {
        response = await fetch(`${API_BASE}/api/denoise/sample`, { method: "POST" });
      } else if (selectedFile) {
        const form = new FormData();
        form.append("file", selectedFile);
        response = await fetch(`${API_BASE}/api/denoise`, { method: "POST", body: form });
      } else if (pasteInput.value.trim()) {
        const payload = parsePastedText(pasteInput.value);
        response = await fetch(`${API_BASE}/api/denoise`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } else {
        setStatus("Add a file, paste data, or choose the demo spectrum first.", "error");
        setRunning(false);
        return;
      }

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || `Request failed (${response.status}).`);
      }

      lastResult = data;
      renderResults(data);
      setStatus(`Done — ${data.x.length.toLocaleString()} points processed.`, "success");
      showToast("Denoising complete.");
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      console.error(err);
      const isNetworkErr = err instanceof TypeError;
      setStatus(
        isNetworkErr
          ? "Could not reach the Spectrapur backend. Check that the API is running and reachable."
          : err.message || "Something went wrong while denoising.",
        "error"
      );
    } finally {
      setRunning(false);
    }
  }

  runBtn.addEventListener("click", runDenoise);

  // ---- Rendering ----
  function renderResults(data) {
    resultsSection.classList.add("visible");
    renderStats(data.stats);
    renderCharts(data);
    renderPeakTable(data.peaks_denoised);
  }

  function renderStats(stats) {
    const cards = [
      {
        label: "Noise reduced",
        value: `${fmt(stats.noise_reduction_pct, 1)}%`,
        positive: stats.noise_reduction_pct > 0,
        sub: `${fmt(stats.noise_std_before, 4)} → ${fmt(stats.noise_std_after, 4)}`,
      },
      {
        label: "Estimated SNR (before → after)",
        value: `${fmt(stats.snr_db_before, 1)} → ${fmt(stats.snr_db_after, 1)} dB`,
        sub: "Signal power vs. high-frequency residual",
      },
      {
        label: "Baseline drift (before → after)",
        value: `${fmt(stats.baseline_drift_before, 3)} → ${fmt(stats.baseline_drift_after, 3)}`,
        sub: "End-level minus start-level offset",
      },
      {
        label: "Max correction applied",
        value: fmt(stats.max_abs_correction, 4),
        sub: "Largest single-point adjustment",
      },
      {
        label: "Data points",
        value: stats.n_points.toLocaleString(),
        sub: "Resampled to 1868 pts for inference",
      },
    ];

    statGrid.innerHTML = cards
      .map(
        (c) => `
      <div class="stat-card">
        <div class="stat-label">${c.label}</div>
        <div class="stat-value ${c.positive ? "positive" : ""}">${c.value}</div>
        <div class="stat-sub">${c.sub}</div>
      </div>`
      )
      .join("");
  }

  function downsampleForChart(x, y1, y2, maxPoints = 1500) {
    const n = x.length;
    if (n <= maxPoints) return { x, y1, y2 };
    const step = Math.ceil(n / maxPoints);
    const rx = [], ry1 = [], ry2 = [];
    for (let i = 0; i < n; i += step) {
      rx.push(x[i]);
      ry1.push(y1[i]);
      if (y2) ry2.push(y2[i]);
    }
    return { x: rx, y1: ry1, y2: y2 ? ry2 : null };
  }

  function renderCharts(data) {
    function renderCharts(data) {
    if (typeof Chart === "undefined") {
      document.querySelectorAll(".chart-wrap").forEach((el) => {
        el.innerHTML =
          '<p class="empty-note" style="padding-top:20px;">Charting library failed to load from the CDN. Check your network/ad-blocker, then refresh and re-run.</p>';
      });
      return;
    }
    const { x, y1: yNoisy, y2: yDenoised } = downsampleForChart(data.x, data.y_noisy, data.y_denoised);
    const { y1: yResidual } = downsampleForChart(data.x, data.residual, null);

    const overlayCtx = document.getElementById("overlay-chart").getContext("2d");
    const residualCtx = document.getElementById("residual-chart").getContext("2d");

    if (overlayChart) overlayChart.destroy();
    if (residualChart) residualChart.destroy();

    const sharedScales = {
      x: {
        type: "linear",
        title: { display: true, text: "x", color: "#4d5f7a" },
        grid: { color: "#eef5fb" },
        ticks: { color: "#4d5f7a" },
      },
      y: {
        title: { display: true, text: "intensity", color: "#4d5f7a" },
        grid: { color: "#eef5fb" },
        ticks: { color: "#4d5f7a" },
      },
    };

    overlayChart = new Chart(overlayCtx, {
      type: "line",
      data: {
        labels: x,
        datasets: [
          {
            label: "Raw",
            data: yNoisy,
            borderColor: "#A9CCE8",
            backgroundColor: "transparent",
            borderWidth: 1.4,
            pointRadius: 0,
            tension: 0.15,
          },
          {
            label: "Denoised",
            data: yDenoised,
            borderColor: "#10233F",
            backgroundColor: "transparent",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: false }, tooltip: { enabled: true } },
        scales: sharedScales,
      },
    });

    residualChart = new Chart(residualCtx, {
      type: "line",
      data: {
        labels: x,
        datasets: [
          {
            label: "Raw − Denoised",
            data: yResidual,
            borderColor: "#B3663F",
            backgroundColor: "rgba(179, 102, 63, 0.12)",
            borderWidth: 1.2,
            pointRadius: 0,
            fill: true,
            tension: 0.15,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false } },
        scales: sharedScales,
      },
    });
  }

  function renderPeakTable(peaks) {
    if (!peaks || peaks.length === 0) {
      peakTableBody.innerHTML = `<tr><td colspan="3" class="empty-note">No prominent peaks detected.</td></tr>`;
      return;
    }
    peakTableBody.innerHTML = peaks
      .map(
        (p, i) => `<tr><td>${i + 1}</td><td>${fmt(p.x, 2)}</td><td>${fmt(p.y, 4)}</td></tr>`
      )
      .join("");
  }

  // ---- Downloads ----
  downloadCsvBtn.addEventListener("click", () => {
    if (!lastResult) return;
    const { x, y_noisy, y_denoised } = lastResult;
    let csv = "x,y_raw,y_denoised\n";
    for (let i = 0; i < x.length; i++) {
      csv += `${x[i]},${y_noisy[i]},${y_denoised[i]}\n`;
    }
    downloadBlob(csv, "spectrapur_denoised.csv", "text/csv");
  });

  downloadReportBtn.addEventListener("click", () => {
    if (!lastResult) return;
    const report = {
      generated_by: "Spectrapur",
      model: "ResUNet1D (nabhya8013/spectral_denoise)",
      stats: lastResult.stats,
      peaks_noisy: lastResult.peaks_noisy,
      peaks_denoised: lastResult.peaks_denoised,
    };
    downloadBlob(JSON.stringify(report, null, 2), "spectrapur_report.json", "application/json");
  });

  downloadPngBtn.addEventListener("click", () => {
    if (!overlayChart) return;
    const link = document.createElement("a");
    link.href = overlayChart.toBase64Image();
    link.download = "spectrapur_chart.png";
    link.click();
  });

  function downloadBlob(content, filename, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }
})();
