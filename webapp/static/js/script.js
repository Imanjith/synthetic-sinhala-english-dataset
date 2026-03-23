document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyzeBtn");
  const inputField = document.getElementById("sinhalaInput");
  const resultCard = document.getElementById("resultCard");
  const riskBadge = document.getElementById("riskBadge");
  const translationText = document.getElementById("translationText");
  const logProbVal = document.getElementById("logProbVal");
  const isHallucVal = document.getElementById("isHallucVal");
  const tokenCountVal = document.getElementById("tokenCountVal");
  const labseScoreVal = document.getElementById("labseScoreVal");

  analyzeBtn.addEventListener("click", async () => {
    const text = inputField.value.trim();
    if (!text) return;

    // UI Loading State
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('span').textContent = "Analysing…";
    resultCard.classList.add("hidden");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error("Analysis failed");
      }

      const data = await response.json();
      displayResults(data);
    } catch (error) {
      console.error(error);
      alert("An error occurred during analysis. Please try again.");
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.querySelector('span').textContent = "Translate & Analyse →";
    }
  });

  function displayResults(data) {
    // Update Translation
    translationText.textContent = data.translation;

    // Update Metrics
    logProbVal.textContent = data.log_prob.toFixed(4);
    
    // Is Hallucinated? 
    isHallucVal.textContent = data.is_hallucinated ? "Yes" : "No";
    isHallucVal.style.color = data.is_hallucinated ? "var(--accent-red)" : "var(--accent-green)";

    tokenCountVal.textContent = (data.hallucination_risk * 100).toFixed(1) + "%";

    // LaBSE Score
    if (data.labse_score !== undefined) {
      const pct = (data.labse_score * 100).toFixed(1);
      labseScoreVal.textContent = pct + "%";
      if (data.labse_score < 0.4) {
        labseScoreVal.style.color = "var(--accent-red)";
      } else if (data.labse_score > 0.7) {
        labseScoreVal.style.color = "var(--accent-green)";
      } else {
        labseScoreVal.style.color = "var(--ink)";
      }
    }

    // Update Risk Badge
    riskBadge.className = "badge"; // Reset classes
    riskBadge.textContent = data.risk_level;

    if (data.risk_level.includes("Safe") || data.risk_level.includes("Low")) {
      riskBadge.classList.add("safe");
    } else if (data.risk_level.includes("Medium")) {
      riskBadge.classList.add("medium");
    } else if (data.risk_level.includes("High")) {
      riskBadge.classList.add("high");
    } else {
      riskBadge.classList.add("critical");
    }

    // Show Result Card
    resultCard.classList.remove("hidden");
  }
});
