# Hallucination Detection and Mitigation for Sinhala-to-English NMT

A comprehensive research tool, dataset, and framework designed for detecting and mitigating "pathological" hallucinations in Sinhala-to-English Neural Machine Translation (NMT) outputs. This project was developed as part of a final-year research thesis to ensure NMT safety in low-resource deployment scenarios.

## Overview

Machine translation for low-resource languages like Sinhala is highly susceptible to generating fluent but entirely incorrect translations, a phenomenon known as a "hallucination". In low-resource settings, weak cross-lingual alignment often leads to these severe semantic disconnects. Standard sequence-to-sequence metrics like BLEU or chrF capture surface-level textual overlaps but often fail to penalize model hallucinations where the output completely diverges from the source meaning while remaining grammatically fluent.

This project tackles this issue by employing a rigorous, reference-free three-signal ensemble approach to evaluate translation reliability. Rather than relying on a single metric, the system triangulates the probability of hallucination by observing neural risk scores, the model's intrinsic uncertainty, and cross-lingual semantic alignment.

## Methodology & Mathematical Formulation

The reference-free hallucination detection framework is based on three core metric layers. Below are the in-depth mechanics and equations governing each signal:

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/9e24b3f7-2569-4e17-9ffe-5923feae1859" />

### 1. Token-Level Risk (mDeBERTa-v3)
We fine-tuned `microsoft/mdeberta-v3-base` for token-level sequence labelling. This enables precise error localisation by accepting the source sentence and the generated English translation to classify each token in the hypothesis as either "safe" (0) or "hallucinated" (1). 

- **Training**: The model was trained dynamically via sequence matching against our aligned synthetic dataset. Crucially, we introduced a novel **semantic rescue mechanism** that utilises character-level similarity to distinguish genuine semantic hallucinations from valid lexical paraphrases.
- **Inference Computation**: The system calculates the fractional risk of the sentence based on the token predictions. Let $N$ be the total number of hypothesis tokens, and $\hat{y}_i$ be the boolean prediction for the $i$-th token.

$$ \text{Token Risk (mDeBERTa)} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\hat{y}_i = 1) $$

### 2. Sequence Log Probability (Teacher-Forcing Uncertainty)
Sequence log probability acts as the NMT model's internal confidence parameter. Instead of relying on beam search decoding logits (which vary heavily across model architectures like NLLB vs M2M100), it is computed via a **teacher-forcing forward pass** acting on the generated sequence. Lower log probabilities mathematically reflect high prediction uncertainty at the decoding level.

Given a source sequence $X$ and a generated hypothesis sequence $Y$, the log probability is derived from the cross-entropy loss function of the auto-regressive NMT model $\theta$:

$$ \text{Seq LogProb} = -\mathcal{L}_{\text{CE}} \left( Y \mid X, \theta \right) = \frac{1}{|Y|} \sum_{t=1}^{|Y|} \log P(y_t \mid y_{<t}, X, \theta) $$

This negative log-likelihood acts as a strong, standardized heuristic layer for underlying semantic hallucinations across different NMT model families.

### 3. Cross-Lingual Semantic Similarity (LaBSE)
To verify meaning preservation regardless of arbitrary text decoding paths, we utilize Language-Agnostic BERT Sentence Embeddings (LaBSE). This module computes cross-lingual vector embeddings for both the Sinhala source ($E_{src}$) and the English hypothesis ($E_{hyp}$) via mean-pooling across non-padding tokens. By measuring the cosine distance between the L2-normalized embeddings, we establish a rigid threshold to ensure concepts remain strictly aligned.

$$ \text{LaBSE Similarity} = \frac{E_{src} \cdot E_{hyp}}{\|E_{src}\| \|E_{hyp}\|} = \cos(\theta) $$

## Ensemble Verdict Logic

The web application aggregates these metrics to provide an automated, reliable, and highly actionable verdict based on precise boundaries:
- **High Risk**: Both token-level neural risk scores (mDeBERTa) and sequence metrics (log-prob) fall outside established safety thresholds.
- **Medium Risk**: A divergence occurs where only one of the primary signals indicates a potential hallucination boundary.
- **Safe**: All signals safely align, confirming high likelihood of translation fidelity.

Extensive benchmarking across **eight diverse NMT systems** has proven that this triple-signal approach effectively identifies semantic disconnects that traditional confidence-based metrics overlook, showing strong Spearmean correlations with advanced established metrics such as BERTScore and COMET.

**Visual Dashboard Metrics:**
![Scanner Web Interface](<img width="881" height="613" alt="image" src="https://github.com/user-attachments/assets/bfa4dc7b-bc0b-4758-ad4d-81a9a061788c" />)
<img width="970" height="798" alt="image" src="https://github.com/user-attachments/assets/1a7b33ce-bcc6-46ee-a61e-e2791f1bf0b6" />
<img width="899" height="822" alt="image" src="https://github.com/user-attachments/assets/4bd9d8b8-1980-4e98-b0ef-7319b702e27d" />
<img width="920" height="673" alt="image" src="https://github.com/user-attachments/assets/0e3b2185-6386-447b-8062-84c5d784cfea" />


## Synthetic Dataset Generation

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/ff020fd8-ba90-43f9-9e16-3d2afb0499fb" />

Training the localized mDeBERTa token classifier required a carefully curated, balanced dataset. Rather than relying on computationally expensive human annotation, we introduced a synthetic dataset of **45,000 sentence pairs** generated through a probabilistic chain of five linguistically motivated corruption strategies. The automated pipeline performs:

1. **Model Distillation & Corruption**: Utilizes models like NLLB 1.3B paired with highly stochastic generation temperatures and probabilistic corruption chains (e.g., Entity Swapping via spaCy) to force semantic divergence away from grounded source texts.
2. **Grammar Validity Filtering**: Employs an offline LanguageTool wrapper to specifically filter out poor generations containing broken grammar. This isolation strictly forces the downstream detector to learn semantic discrepancies rather than identifying easily detectable typographical errors.
3. **Class Balancing & Semantic Rescue**: Utilizes character-level sequence matching to prevent paraphrasing from being marked as hallucinated. It then downsamples the baseline sets to guarantee an exact 50-50 class split between valid translations and generated hallucinated strings, eliminating fundamental classifier bias.

## Project Structure

```text
├── Datasets/                               # 45,000-pair synthetic datasets (e.g., balanced 15k sets)
├── models/ / saved_model/                  # Fine-tuned mDeBERTa token-classification checkpoints
├── webapp/                                 # Interactive hallucination scanner web interface
│   ├── app.py                              # FastAPI backend aggregating M2M100 + mDeBERTa + LaBSE
│   ├── templates/index.html                # Frontend UI utilizing dynamic risk logic gauges
│   └── static/                             # CSS styling and functional JavaScript state management
├── Dataset Generation.ipynb                # Probabilistic data generation and grammar filtering pipeline
├── Main Hallucination Detector.ipynb       # Detector training loops and baseline evaluation scripts
└── Hallucination Detector Comparison.ipynb # Quantitative evaluations across 8 NMT systems vs BERTScore/COMET
```

## Setup and Installation

The web application acts as local testbed, exposing an inference API and a premium dashboard visualizing the triple-signal evaluation logic against a live M2M100 (418M) machine translation pass.

### 1. Install Dependencies
```bash
pip install fastapi uvicorn torch transformers sentence-transformers numpy pydantic
```

### 2. Start the Application Server
Navigate to the root directory of the project and execute:
```bash
python webapp/app.py
```
*Note: Bootstrapping initializes HuggingFace downloads of model weights (M2M-418M, mDeBERTa-v3, and LaBSE) upon first run.*

### 3. Usage Evaluation
Navigate to `http://localhost:8000` via your web browser. Input a Sinhala text sequence. The system will perform real-time translation and visualize dynamically scaled gauge thresholds mapping the log probability margin, semantic alignment distance, and mDeBERTa fractional risk.

## Academic Context
Developed and configured for academic NMT evaluation research contexts (BSc. Hons in AI and Data Science thesis). Always consult individual repository licensing domains for the integration of upstream HuggingFace model architectures before any enterprise implementations.
