# Sinhala-to-English NMT Hallucination Scanner

A comprehensive research tool and pipeline for detecting hallucinations in Sinhala-to-English Neural Machine Translation (NMT) outputs. 

## Overview

Machine translation for low-resource languages like Sinhala is highly susceptible to generating fluent but entirely incorrect translations, a phenomenon known as a "hallucination". Standard sequence-to-sequence metrics like BLEU or chrF capture surface-level textual overlaps but often fail to penalize model hallucinations where the output completely diverges from the source meaning while remaining grammatically fluent.

This project tackles this issue by employing a rigorous three-signal ensemble approach to evaluate translation reliability. Rather than relying on a single metric, the system triangulates the probability of hallucination by observing the model's intrinsic uncertainty, token-specific structural risks, and cross-lingual semantic alignment.

## Methodology

The hallucination detection logic is based on the following three core metric layers:

### 1. Token-Level Risk (mDeBERTa)
We fine-tuned `microsoft/mdeberta-v3-base` as a sequence tagger. It accepts the source sentence and the generated English translation to classify each token in the hypothesis as either "safe" or "hallucinated". 
- The model was trained dynamically via sequence matching against an aligned synthetic dataset, penalizing ungrounded insertions and non-paraphrased replacements.
- During inference, the system calculates the percentage of flagged tokens to determine a base risk factor. 

### 2. Sequence Log Probability
Sequence log probability acts as the NMT model's internal confidence parameter. It is computed via a teacher-forcing forward pass acting on the generated sequence. Lower log probabilities mathematically reflect high prediction uncertainty at the decoding level, which acts as a strong heuristic layer for underlying semantic hallucinations.

### 3. Semantic Similarity (LaBSE)
To verify meaning preservation regardless of arbitrary text decoding paths, we utilize Language-Agnostic BERT Sentence Embeddings (LaBSE). This module computes cross-lingual vector embeddings for both the Sinhala source and the English hypothesis. By measuring the cosine distance between them, we establish a rigid threshold to ensure concepts remain strictly aligned.

## Ensemble Verdict Logic

The web application aggregates these metrics to provide an automated and highly actionable verdict:
- **High Risk**: Both token-level risk (mDeBERTa) and sequence metrics (log-prob) fall outside established safety thresholds.
- **Medium Risk**: A divergence occurs where only one of the primary signals indicates a potential hallucination boundary.
- **Safe**: All signals suggest high reliability.

![Scanner Web Interface](docs/scanner_interface.png)
*(Note: Please place a screenshot of the interactive web dashboard with confidence gauges active into `docs/scanner_interface.png`)*

## Synthetic Dataset Generation

Training the localized mDeBERTa token classifier required a carefully curated, balanced dataset of faithful versus hallucinated translations. We built an automated pipeline using `Dataset Generation.ipynb` which:
1. **Model Distillation**: Utilizes NLLB 1.3B paired with highly stochastic generation temperatures to force semantic divergence away from grounded source texts.
2. **Grammar Filtering**: Employs an offline LanguageTool wrapper to specifically filter out poor generations containing broken grammar. This isolation strictly forces the downstream detector to learn semantic discrepancies rather than identifying easily detectable typographical errors.
3. **Class Balancing**: Downsamples the baseline sets to guarantee an exact 50-50 class split between valid translations and generated hallucinated strings, eliminating fundamental classifier bias.

![Dataset Architecture Overview](docs/dataset_architecture.png)
*(Note: Please attach a visualization of dataset generation paths or metrics into `docs/dataset_architecture.png`)*

## Project Structure

```text
├── Datasets/                               # Generated synthetic text datasets (e.g., balanced 15k sets)
├── models/ / saved_model/                  # Fine-tuned token-classification checkpoints
├── webapp/                                 # Interactive hallucination scanner web interface
│   ├── app.py                              # FastAPI backend aggregating M2M100 + mDeBERTa + LaBSE
│   ├── templates/index.html                # Frontend UI utilizing dynamic risk logic gauges
│   └── static/                             # CSS styling and functional JavaScript state management
├── Dataset Generation.ipynb                # Data generation and heuristic grammar filtering pipeline
├── Main Hallucination Detector.ipynb       # Detector training loops and baseline evaluation scripts
└── Hallucination Detector Comparison.ipynb # Quantitative evaluations of varying threshold weights
```

## Setup and Installation

The web application acts as local testbed, exposing an inference API and a premium dashboard visualizing the multi-signal evaluation logic against a live M2M100 (418M) machine translation pass.

### 1. Install Dependencies
```bash
pip install fastapi uvicorn torch transformers sentence-transformers numpy pydantic
```

### 2. Start the Application Server
Navigate to the root directory of the project and execute:
```bash
python webapp/app.py
```
*Note: Bootstrapping initializes HuggingFace downloads of model weights upon first run.*

### 3. Usage Evaluation
Navigate to `http://localhost:8000` via your web browser. Input a Sinhala text sequence. The system will perform real-time translation and visualize dynamically scaled gauge thresholds mapping the log probability margin, semantic alignment distance, and mDeBERTa fractional risk.

## Licensing
Configured for academic NMT evaluation research contexts. Always consult individual repository licensing domains for the integration of upstream HuggingFace model architectures (M2M100, mDeBERTa, LaBSE) before any enterprise implementations.
