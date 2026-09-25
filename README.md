# Fact over Fiction: Hallucination Detection for Sinhala-to-English NMT

Code, data and results for reference-free, token-level hallucination detection in Sinhala→English neural machine translation.

Machine translation for low-resource languages like Sinhala can produce fluent English that is unrelated to the source. This repository contains:

- a **45,000-row synthetic hallucination corpus** built from five linguistically motivated corruption strategies,
- a fine-tuned **mDeBERTa-v3 token-level detector** (token F1 0.841 ± 0.001 over three seeds on a source-disjoint test set),
- a **source-ablation control** showing that the detector relies on the Sinhala source rather than on surface artefacts of the corruption process,
- an evaluation of a **three-signal ensemble** (detector risk, sequence log-probability, LaBSE similarity), and
- a **benchmark of eight Sinhala→English NMT systems**.

Every number in the paper can be traced to a cell output in the notebooks or to a file in `Results/`.

## Repository structure

```text
├── Dataset Generation.ipynb          # Corpus generation, grammar filter, balancing (with outputs)
├── Main Hallucination Detector.ipynb # Training, evaluation, source ablation, ensemble, benchmark (with outputs)
├── Datasets/
│   ├── synthetic_hallucinations_full.csv      # 45,000-row released corpus
│   └── synthetic_hallucinations_balanced.csv  # 15,000-row balanced subset used for all experiments
├── results/                          # Splits, result tables, benchmark outputs, calibrated thresholds
└── webapp/                           # Demo scanner interface (earlier version, see note below)
```

Detector checkpoints (about 1 GB) are hosted separately: **[link to checkpoints]**

## Dataset

Built from `NLPC-UOM/nllb-top25k-ensi-cleaned`. 7,500 source sentences were sampled (seed 42) and five negative samples planned per source, giving 45,000 rows (32,577 hallucinated, 12,423 faithful).

| Strategy | What it does |
|---|---|
| High-temperature sampling | NLLB-200-1.3B at T = 1.5 (top-p 0.95, top-k 50); kept only if BERTScore F1 < 0.92 |
| Entity (NER) swap | Replaces one entity with a same-label entity from a pool |
| Semantic drift | Replaces one non-entity content word with a WordNet antonym |
| Dependency swap | Swaps the subject and object of the same verb |
| Numeric distortion | Increments, decrements or appends a digit to one number |

A probabilistic chain applies a second strategy to 80% of hallucinated rows, producing compound hallucinations (the `method` column records the chain, e.g. `temp + ner`). A LanguageTool grammar filter is applied before balancing. The balanced subset uses the 7,500 references as faithful samples and 7,500 randomly drawn hallucinations, split 70/10/20 **by source sentence** so no source appears in more than one split.

**Columns:** `sinhala`, `hypothesis`, `reference`, `label` (1 = hallucinated), `method`.

## Detector

`microsoft/mdeberta-v3-base` fine-tuned for token classification on `[CLS] source [SEP] hypothesis [SEP]`. Token labels come from `difflib` alignment with the reference, plus a semantic-rescue step (Ratcliff/Obershelp similarity ≥ 0.80 against the aligned reference span) applied to high-temperature rows. A sentence is flagged if any hypothesis token has P(hallucinated) > 0.5 (τ = 0, calibrated on validation).

| Model | Token F1 (test, 3 seeds) |
|---|---|
| mDeBERTa-v3 | 0.841 ± 0.001 |
| XLM-RoBERTa | 0.789 ± 0.005 |

Source-ablation control: shuffling or removing the source drops sentence-level AUROC from 0.970 to chance, and faithful hypotheses paired with a mismatched source are flagged 100% of the time (4.5% with the correct source).

## Reproducing the results

Both notebooks were run on Kaggle with a single NVIDIA T4.

1. Run `Dataset Generation.ipynb`. It writes the two CSVs in `Datasets/`.
2. Upload the balanced CSV as a Kaggle dataset and set `DATASET_FILE` in `Main Hallucination Detector.ipynb`.
3. Run `Main Hallucination Detector.ipynb`. The `RUN_*` flags at the top allow resuming across sessions. Outputs (tables, splits, benchmark signals, checkpoints) are written to `/kaggle/working`.

All randomness is seeded (generation and splits: 42; training: 42, 43, 44).

## Web demo

`webapp/` contains a FastAPI demo that translates Sinhala input with M2M-100 (418M) and displays detector risk, log-probability and LaBSE similarity. **It predates the revised experiments** and uses an earlier detector and thresholds, so its verdicts do not correspond to the results reported in the paper.

```bash
pip install fastapi uvicorn torch transformers sentence-transformers numpy pydantic
python webapp/app.py   # then open http://localhost:8000
```

## Limitations

The detector is trained on synthetic corruptions; its precision on naturally occurring translation errors has not been measured with human annotation. Thresholds were calibrated on a balanced split and are not calibrated for the much lower hallucination rates of real translations. See the paper's Limitations section for details.

## Citation

```bibtex
@inproceedings{obeysekara2026factoverfiction,
  title     = {Fact over Fiction: Detection of Pathological Hallucinations in Sinhala-to-English Neural Machine Translation},
  author    = {Obeysekara, Navam and Jayatilleke, Nevidu},
  booktitle = {Proceedings of ROCLING 2026},
  year      = {2026}
}
```
