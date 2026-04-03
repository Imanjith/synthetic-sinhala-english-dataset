import os
import logging
import torch
import numpy as np
from typing import List
from huggingface_hub import snapshot_download
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    M2M100ForConditionalGeneration,
    AutoModel
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
M2M_MODEL_NAME           = "./models"
HALLUCINATION_MODEL_PATH = "./saved_model"
LABSE_MODEL_PATH         = "./labse_model"

M2M_REPO_ID           = "facebook/m2m100_418M"
HALLUCINATION_REPO_ID = "imanjith/mdbertav3trained"
LABSE_REPO_ID         = "sentence-transformers/LaBSE"

# Risk thresholds (calibrated from notebook results for M2M 418M)
# Avg log-prob for m2m_418m was -1.4839; 30th-pct threshold was -1.6582
# We use a slightly more generous fixed threshold for single-sentence inference.
LOGPROB_THRESHOLD  = -1.70   # below this → low confidence
MDEBERTA_THRESHOLD = 0.05    # fraction of hallucianted tokens (0-1)

device = "cpu"  # Forced to CPU for INT8 Dynamic Quantization support

# ---------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------------------------
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    source_text: str
    translation: str
    log_prob: float
    hallucination_risk: float        # fraction of tokens flagged by mDeBERTa (0-1)
    is_hallucinated: bool            # True if hallucination_risk > MDEBERTA_THRESHOLD
    flagged_tokens: list[str] = []   # list of exact string tokens flagged
    labse_score: float               # semantic similarity across languages (0-1)
    risk_level: str                  # "Low" | "Medium" | "High"

# ---------------------------------------------------------------------------
# AUTO-DOWNLOAD MISSING MODELS
# ---------------------------------------------------------------------------
def ensure_model_exists(repo_id: str, local_dir: str):
    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        logger.info(f"Directory '{local_dir}' is empty/missing. Downloading '{repo_id}' from Hugging Face...")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        logger.info(f"Successfully downloaded '{repo_id}' to '{local_dir}'.")

ensure_model_exists(repo_id=M2M_REPO_ID, local_dir=M2M_MODEL_NAME)
ensure_model_exists(repo_id=HALLUCINATION_REPO_ID, local_dir=HALLUCINATION_MODEL_PATH)
ensure_model_exists(repo_id=LABSE_REPO_ID, local_dir=LABSE_MODEL_PATH)

# ---------------------------------------------------------------------------
# MODEL LOADING — eager, at import time (before uvicorn event loop starts)
# ---------------------------------------------------------------------------
logger.info(f"Loading models on device: {device}")

models = {}

# 1. M2M100 translation model
try:
    logger.info("Loading M2M100 (418M) tokenizer...")
    _tok_m2m = AutoTokenizer.from_pretrained(M2M_MODEL_NAME)
    logger.info("Loading M2M100 (418M) model weights...")
    _mdl_m2m = M2M100ForConditionalGeneration.from_pretrained(M2M_MODEL_NAME)
    logger.info("Compressing M2M100 to INT8...")
    _mdl_m2m = torch.quantization.quantize_dynamic(_mdl_m2m, {torch.nn.Linear}, dtype=torch.qint8)
    _mdl_m2m = _mdl_m2m.to(device)
    _mdl_m2m.eval()
    models["tokenizer_m2m"] = _tok_m2m
    models["model_m2m"]     = _mdl_m2m
    logger.info("M2M100 loaded.")
except Exception as e:
    logger.error(f"ERROR loading M2M100: {e}", exc_info=True)

# 2. mDeBERTa hallucination detector
try:
    logger.info(f"Loading hallucination detector from: {HALLUCINATION_MODEL_PATH} ...")
    _tok_det = AutoTokenizer.from_pretrained(HALLUCINATION_MODEL_PATH)
    _mdl_det = AutoModelForTokenClassification.from_pretrained(HALLUCINATION_MODEL_PATH, num_labels=2)
    logger.info("Compressing Hallucination Detector to INT8...")
    _mdl_det = torch.quantization.quantize_dynamic(_mdl_det, {torch.nn.Linear}, dtype=torch.qint8)
    _mdl_det = _mdl_det.to(device)
    _mdl_det.eval()
    models["tokenizer_halluc"] = _tok_det
    models["model_halluc"]     = _mdl_det
    logger.info("Hallucination detector loaded.")
except Exception as e:
    logger.error(f"ERROR loading hallucination detector: {e}", exc_info=True)

# 3. LaBSE semantic similarity model
try:
    logger.info("Loading LaBSE semantic similarity model...")
    _tok_labse = AutoTokenizer.from_pretrained(LABSE_MODEL_PATH)
    _mdl_labse = AutoModel.from_pretrained(LABSE_MODEL_PATH)
    logger.info("Compressing LaBSE to INT8...")
    _mdl_labse = torch.quantization.quantize_dynamic(_mdl_labse, {torch.nn.Linear}, dtype=torch.qint8)
    _mdl_labse = _mdl_labse.to(device)
    _mdl_labse.eval()
    models["tokenizer_labse"] = _tok_labse
    models["model_labse"]     = _mdl_labse
    logger.info("LaBSE model loaded.")
except Exception as e:
    logger.error(f"ERROR loading LaBSE model: {e}", exc_info=True)

logger.info(f"Startup complete. Loaded models: {list(models.keys())}")


# ---------------------------------------------------------------------------
# INFERENCE HELPERS
# ---------------------------------------------------------------------------

def generate_translation(text: str) -> tuple[str, float]:
    """
    Translate Sinhala text → English using M2M100.
    Returns (translation_string, avg_log_prob).

    Log-prob is computed via a teacher-forcing forward pass on the model's
    own output (same method as the notebook's safe_translate_score).
    Higher (less negative) = more confident.
    """
    tokenizer = models["tokenizer_m2m"]
    model     = models["model_m2m"]

    tokenizer.src_lang = "si"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    forced_bos = tokenizer.get_lang_id("en")

    with torch.no_grad():
        # Generate translation
        gen_ids = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_new_tokens=256,
        )

        # Decode
        translation = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Teacher-forcing pass: feed the generated sequence back as labels
        # to get the model's NLL loss, then negate to get log-prob.
        labels = gen_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        out = model(**inputs, labels=labels)
        log_prob = float(-out.loss.item())   # higher = more confident

    return translation, log_prob


def detect_hallucinations(source_text: str, translation: str) -> tuple[float, List[int]]:
    """
    Run the fine-tuned mDeBERTa model on the (source, translation) pair.

    Input format (same as training in the notebook):
        [CLS] Sinhala_source [SEP] English_hypothesis [SEP]

    Returns:
        risk_fraction  – fraction of hypothesis tokens classified as hallucinated (0-1)
        flagged_indices – list of hypothesis token positions that were flagged
    """
    tokenizer = models["tokenizer_halluc"]
    model     = models["model_halluc"]

    # Tokenize as a pair — the tokenizer automatically inserts [CLS]…[SEP]…[SEP]
    enc = tokenizer(
        source_text,
        translation,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits   # shape: (1, seq_len, 2)

    preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()  # (seq_len,)

    # Locate the two [SEP] tokens so we can isolate the hypothesis segment
    input_ids  = enc["input_ids"].squeeze(0).cpu().numpy()
    sep_id     = tokenizer.sep_token_id
    sep_positions = [i for i, tok in enumerate(input_ids) if tok == sep_id]

    if len(sep_positions) >= 2:
        hyp_start = sep_positions[0] + 1   # first token after first [SEP]
        hyp_end   = sep_positions[1]        # up to (not including) second [SEP]
        hyp_preds = preds[hyp_start:hyp_end]
        hyp_ids   = input_ids[hyp_start:hyp_end]
    else:
        # Fallback: use everything after position 1 (skip [CLS])
        hyp_preds = preds[1:]
        hyp_ids   = input_ids[1:]

    flagged_indices = [int(i) for i, p in enumerate(hyp_preds) if p == 1]
    
    # Extract the exact string tokens that were flagged
    flagged_tokens = []
    for idx in flagged_indices:
        tok_str = tokenizer.decode([hyp_ids[idx]])
        # Clean mDeBERTa prefix and whitespace
        cleaned = tok_str.replace(" ", "").strip()
        # Avoid short punctuation artifacts
        if len(cleaned) > 1 and cleaned not in flagged_tokens:
            flagged_tokens.append(cleaned)

    risk_fraction   = float(np.mean(hyp_preds == 1)) if len(hyp_preds) > 0 else 0.0

    return risk_fraction, flagged_tokens


def compute_labse_similarity(source_text: str, translation: str) -> float:
    """
    Computes cosine similarity between source and translation using LaBSE.
    """
    tokenizer = models["tokenizer_labse"]
    model     = models["model_labse"]
    
    encoded_input = tokenizer(
        [source_text, translation],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
        
    # Mean pooling
    attention_mask = encoded_input['attention_mask']
    token_embeddings = model_output[0]
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    
    # L2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    sim = torch.sum(embeddings[0] * embeddings[1]).item()
    return float(sim)


def categorize_risk(log_prob: float, mdeberta_risk: float) -> str:
    """
    Ensemble the two signals into a single risk label.

    Rules (mirrors the notebook's ensemble logic):
      - High   : both signals flag it
      - Medium : either signal flags it
      - Low    : neither signal flags it
    """
    low_confidence = log_prob      < LOGPROB_THRESHOLD
    high_risk      = mdeberta_risk > MDEBERTA_THRESHOLD

    if low_confidence and high_risk:
        return "High"
    elif low_confidence or high_risk:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

@app.get("/")
def home():
    return FileResponse("webapp/templates/index.html")


@app.post("/predict", response_model=TranslationResponse)
async def predict(request: TranslationRequest):
    if "model_m2m" not in models or "model_halluc" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    # 1. Translate + compute log-prob confidence
    translation, log_prob = generate_translation(text)
    
    # --- DEBUG / TESTING FEATURE ---
    # If the user prefixes the text with "TEST:", force a complete hallucination
    # so they can see the mDeBERTa model flag it correctly.
    if text.startswith("TEST:"):
        text = text.replace("TEST:", "").strip()
        translation = "A random completely hallucinated sentence that means nothing in Sinhala."
        log_prob = -5.0 # Low confidence
    # -------------------------------

    # 2. Detect hallucinations with mDeBERTa
    halluc_risk, flagged_tokens = detect_hallucinations(text, translation)

    # 3. Compute LaBSE Similarity
    labse_score = compute_labse_similarity(text, translation)

    # 4. Ensemble → risk label
    risk = categorize_risk(log_prob, halluc_risk)
    
    # 5. Binary hallucination flag (like notebook)
    is_halluc = halluc_risk > MDEBERTA_THRESHOLD

    return TranslationResponse(
        source_text=text,
        translation=translation,
        log_prob=round(log_prob, 4),
        hallucination_risk=round(halluc_risk, 4),
        is_hallucinated=is_halluc,
        flagged_tokens=flagged_tokens,
        labse_score=round(labse_score, 4),
        risk_level=risk,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
