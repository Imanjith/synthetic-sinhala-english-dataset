// ── Thresholds (mirror app.py constants) ──
const LOGPROB_THRESHOLD  = -1.70;   // below → low confidence
const MDEBERTA_THRESHOLD = 0.20;    // fraction; above → high risk
const LABSE_LOW          = 0.40;    // below → poor semantic match
const LABSE_HIGH         = 0.70;    // above → good semantic match

// Map a log-prob value to a 0–100 pct for the gauge.
// Practical range observed: −5.0 (very bad) → 0.0 (perfect).
function logProbToPct(lp) {
  const min = -5.0, max = 0.0;
  return Math.max(0, Math.min(100, ((lp - min) / (max - min)) * 100));
}

document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn      = document.getElementById("analyzeBtn");
  const inputField      = document.getElementById("sinhalaInput");
  const resultCard      = document.getElementById("resultCard");
  const riskBadge       = document.getElementById("riskBadge");
  const translationText = document.getElementById("translationText");

  // Metric value spans
  const logProbVal   = document.getElementById("logProbVal");
  const isHallucVal  = document.getElementById("isHallucVal");
  const tokenCountVal = document.getElementById("tokenCountVal");
  const labseScoreVal = document.getElementById("labseScoreVal");

  // Sub-labels
  const logProbSub  = document.getElementById("logProbSub");
  const riskSub     = document.getElementById("riskSub");
  const labseSub    = document.getElementById("labseSub");
  const hallucSub   = document.getElementById("hallucSub");

  // Gauges
  const logProbGauge    = document.getElementById("logProbGauge");
  const labseGauge      = document.getElementById("labseGauge");
  const riskGauge       = document.getElementById("riskGauge");
  const logProbGaugePct = document.getElementById("logProbGaugePct");
  const labseGaugePct   = document.getElementById("labseGaugePct");
  const riskGaugePct    = document.getElementById("riskGaugePct");

  // Signal verdicts
  const sigLogProbVerdict  = document.getElementById("sigLogProbVerdict");
  const sigMDeBERTaVerdict = document.getElementById("sigMDeBERTaVerdict");
  const sigLaBSEVerdict    = document.getElementById("sigLaBSEVerdict");
  const sigEnsembleVerdict = document.getElementById("sigEnsembleVerdict");

  // Flagged tokens
  const flaggedTokensEl = document.getElementById("flaggedTokens");

  // Interpretation
  const interpretationBox  = document.getElementById("interpretationBox");
  const interpretationText = document.getElementById("interpretationText");

  // ── Submit ──
  analyzeBtn.addEventListener("click", async () => {
    const text = inputField.value.trim();
    if (!text) return;

    analyzeBtn.disabled = true;
    analyzeBtn.querySelector("span").textContent = "Analysing…";
    resultCard.classList.add("hidden");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) throw new Error("Analysis failed");

      const data = await response.json();
      displayResults(data);
    } catch (error) {
      console.error(error);
      alert("An error occurred during analysis. Please try again.");
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.querySelector("span").textContent = "Translate & Analyse →";
    }
  });

  // ── Main display function ──
  function displayResults(data) {
    const lp    = data.log_prob;           // e.g. -1.2
    const risk  = data.hallucination_risk; // 0–1
    const labse = data.labse_score;        // 0–1
    const level = data.risk_level;         // "Low" | "Medium" | "High"

    // ── Translation ──
    translationText.textContent = data.translation;

    // ── Primary metric values ──
    logProbVal.textContent  = lp.toFixed(4);
    tokenCountVal.textContent = (risk * 100).toFixed(1) + "%";
    labseScoreVal.textContent = (labse * 100).toFixed(1) + "%";

    // Is Hallucinated
    isHallucVal.textContent = data.is_hallucinated ? "Yes" : "No";
    isHallucVal.style.color = data.is_hallucinated
      ? "var(--accent-red)"
      : "var(--accent-green)";

    // Sub-labels
    logProbSub.textContent  = lp < LOGPROB_THRESHOLD ? "Low confidence" : "Adequate confidence";
    riskSub.textContent     = risk > MDEBERTA_THRESHOLD ? "Above threshold" : "Below threshold";
    labseSub.textContent    = labse < LABSE_LOW ? "Poor match" : labse > LABSE_HIGH ? "Strong match" : "Moderate match";
    hallucSub.textContent   = "threshold: " + (MDEBERTA_THRESHOLD * 100).toFixed(0) + "%";

    // Colour metric values
    logProbVal.style.color = lp < LOGPROB_THRESHOLD ? "var(--danger)" : "var(--success)";
    tokenCountVal.style.color = risk > MDEBERTA_THRESHOLD ? "var(--danger)" : "var(--success)";
    labseScoreVal.style.color = labse < LABSE_LOW ? "var(--danger)" : labse > LABSE_HIGH ? "var(--success)" : "var(--ink)";

    // ── Risk Badge ──
    riskBadge.className = "badge";
    riskBadge.textContent = level;
    if (level === "Low")  riskBadge.classList.add("safe");
    else if (level === "Medium") riskBadge.classList.add("medium");
    else riskBadge.classList.add("high");

    // ── Gauges ──
    // Log-prob gauge (higher = better)
    const lpPct = logProbToPct(lp);
    logProbGauge.style.width = lpPct + "%";
    logProbGaugePct.textContent = lpPct.toFixed(0) + "%";
    logProbGauge.className = "gauge-fill";
    if      (lpPct < 40) logProbGauge.classList.add("gauge-fill--danger");
    else if (lpPct < 65) logProbGauge.classList.add("gauge-fill--warning");

    // LaBSE gauge
    const labsePct = (labse * 100);
    labseGauge.style.width = labsePct + "%";
    labseGaugePct.textContent = labsePct.toFixed(0) + "%";
    labseGauge.className = "gauge-fill";
    if      (labsePct < LABSE_LOW * 100)  labseGauge.classList.add("gauge-fill--danger");
    else if (labsePct < LABSE_HIGH * 100) labseGauge.classList.add("gauge-fill--warning");

    // Risk gauge (higher = worse — gauge colour flipped)
    const riskPct = (risk * 100);
    riskGauge.style.width = riskPct + "%";
    riskGaugePct.textContent = riskPct.toFixed(0) + "%";
    riskGauge.className = "gauge-fill";
    if      (riskPct > MDEBERTA_THRESHOLD * 100) riskGauge.classList.add("gauge-fill--danger");
    else if (riskPct > (MDEBERTA_THRESHOLD * 100) * 0.5) riskGauge.classList.add("gauge-fill--warning");

    // ── Signal breakdown ──
    const sigLP = lp < LOGPROB_THRESHOLD;
    const sigMD = risk > MDEBERTA_THRESHOLD;
    const sigLS = labse < LABSE_LOW;

    setVerdict(sigLogProbVerdict,  sigLP ? "Flagged" : "Clear",  sigLP ? "danger" : "ok");
    setVerdict(sigMDeBERTaVerdict, sigMD ? "Flagged" : "Clear",  sigMD ? "danger" : "ok");
    setVerdict(sigLaBSEVerdict,    sigLS ? "Low sim." : "Good",  sigLS ? "warn" : "ok");

    const numFlagged = [sigLP, sigMD, sigLS].filter(Boolean).length;
    let ensembleText, ensembleClass;
    if      (numFlagged === 0)  { ensembleText = "All clear";    ensembleClass = "ok"; }
    else if (numFlagged === 1)  { ensembleText = "Weak signal";  ensembleClass = "warn"; }
    else if (numFlagged === 2)  { ensembleText = "Likely issue"; ensembleClass = "warn"; }
    else                        { ensembleText = "High risk";    ensembleClass = "danger"; }
    setVerdict(sigEnsembleVerdict, ensembleText, ensembleClass);

    // ── Flagged Tokens ──
    flaggedTokensEl.innerHTML = "";
    const tokens = data.flagged_tokens || [];
    if (tokens.length === 0) {
      flaggedTokensEl.innerHTML = '<span class="no-flags">✓ No tokens flagged by mDeBERTa.</span>';
    } else {
      tokens.forEach((tok, i) => {
        const chip = document.createElement("span");
        chip.className = "token-chip";
        chip.textContent = tok;
        chip.style.animationDelay = (i * 0.05) + "s";
        flaggedTokensEl.appendChild(chip);
      });
    }

    // ── Plain-English Interpretation ──
    interpretationBox.className = "interpretation-box";
    const lines = [];

    // Opener sentence based on risk level
    if (level === "Low") {
      lines.push("The translation appears <strong>reliable</strong>.");
      interpretationBox.classList.add("ok-bg");
    } else if (level === "Medium") {
      lines.push("The translation has a <strong>moderate hallucination risk</strong> — review is recommended.");
      interpretationBox.classList.add("warn-bg");
    } else {
      lines.push("The translation is flagged as <strong>high risk</strong> and may contain hallucinations.");
    }

    // Log-prob detail
    if (sigLP) {
      lines.push(
        `The sequence log-probability is <strong>${lp.toFixed(4)}</strong>, ` +
        `below the calibrated threshold of ${LOGPROB_THRESHOLD}, indicating the model generated this translation with <strong>low confidence</strong>.`
      );
    } else {
      lines.push(
        `The sequence log-probability is <strong>${lp.toFixed(4)}</strong>, within the acceptable range, ` +
        `suggesting <strong>adequate generation confidence</strong>.`
      );
    }

    // mDeBERTa detail
    if (sigMD) {
      lines.push(
        `mDeBERTa flagged <strong>${(risk * 100).toFixed(1)}%</strong> of hypothesis tokens as hallucinated ` +
        `(threshold: ${(MDEBERTA_THRESHOLD * 100).toFixed(0)}%)` +
        (tokens.length > 0 ? `, highlighting: <em>${tokens.join(", ")}</em>.` : ".")
      );
    } else {
      lines.push(
        `mDeBERTa token risk is <strong>${(risk * 100).toFixed(1)}%</strong>, well below the ${(MDEBERTA_THRESHOLD * 100).toFixed(0)}% threshold.`
      );
    }

    // LaBSE detail
    if (sigLS) {
      lines.push(
        `The LaBSE semantic similarity score is <strong>${(labse * 100).toFixed(1)}%</strong>, which is low — ` +
        `the translation embedding has drifted significantly from the source, a strong hallucination indicator.`
      );
    } else {
      lines.push(
        `LaBSE semantic similarity is <strong>${(labse * 100).toFixed(1)}%</strong>, ` +
        (labse > LABSE_HIGH ? "confirming <strong>strong semantic alignment</strong> between source and translation." : "suggesting adequate meaning preservation.")
      );
    }

    interpretationText.innerHTML = lines.join(" ");

    // ── Reveal ──
    resultCard.classList.remove("hidden");
  }

  function setVerdict(el, text, cls) {
    el.textContent = text;
    el.className = "signal-verdict " + cls;
  }
});
