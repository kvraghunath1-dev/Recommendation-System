# olg-reco-container/recommender.py
import os, json, math, re, logging
import joblib
import numpy as np

# ---------- Logging setup ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
# In Lambda, the root handler exists; just set level.
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# ---------- Paths ----------
MODEL_PATH = os.getenv("MODEL_PATH", "/var/task/artifacts/model.joblib")
U2I_PATH   = os.getenv("USER2IDX_PATH", "/var/task/artifacts/user2idx.json")
I2I_PATH   = os.getenv("ITEM2IDX_PATH", "/var/task/artifacts/item2idx.json")
E2I_PATH   = os.getenv("EVENT2IDX_PATH","/var/task/artifacts/event2idx.json")

def _load_json(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            logger.info("Loaded mapping %s (size=%d)", path, len(data) if hasattr(data, "__len__") else -1)
            return data
    except FileNotFoundError:
        logger.warning("Mapping %s not found; proceeding without it.", path)
        return {}
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e, exc_info=True)
        return {}

# ---------- Load model + mappings (once per cold start) ----------
logger.info("Loading model from %s ...", MODEL_PATH)
model = joblib.load(MODEL_PATH)
USER2IDX  = _load_json(U2I_PATH)   # optional
ITEM2IDX  = _load_json(I2I_PATH)   # optional
EVENT2IDX = _load_json(E2I_PATH)   # optional

# Log factor shapes if present (BPR models)
uf = getattr(model, "user_factors", None)
itf = getattr(model, "item_factors", None)
if uf is not None and itf is not None:
    logger.info("Model factors: user_factors=%s, item_factors=%s", getattr(uf, "shape", None), getattr(itf, "shape", None))
else:
    logger.warning("Model does not expose user_factors/item_factors; is this an implicit BPR model?")

# ---------- Helpers ----------


def _norm_event(desc: str) -> str:
    if not isinstance(desc, str): return "UNKNOWN"
    s = desc.upper()
    s = s.replace("##", "@")
    s = s.replace(" VS ", " @ ").replace(" V ", " @ ").replace(" AT ", " @ ")
    s = re.sub(r"\s*@\s*", " @ ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _user_idx(mask_id):
    # Try mapping first; then assume incoming is already an internal index
    if USER2IDX:
        u = USER2IDX.get(str(mask_id))
        if u is None and isinstance(mask_id, (int, str)):
            try:
                u = USER2IDX.get(int(mask_id))
            except Exception:
                pass
        return u
    try:
        return int(mask_id)
    except Exception:
        return None

def _item_idx(row):
    """
    Resolve item index with multiple fallbacks:
      1) if iid present and in item2idx -> use it
      2) else if event_description present and in event2idx -> use it
      3) else -> None (unmapped)
    """
    # 1) iid path
    iid = row.get("iid", None)
    if iid is not None and ITEM2IDX:
        key = str(iid)
        # handle float-ish strings like "214.0"
        if key.endswith(".0"):
            try: key = str(int(float(key)))
            except Exception: pass
        idx = ITEM2IDX.get(key)
        if idx is not None:
            return int(idx)
    # 2) fallback to text
    desc = row.get("event_description")
    if desc and EVENT2IDX:
        idx = EVENT2IDX.get(_norm_event(desc))
        if idx is not None: return int(idx)
    # 3) give up
    return None

# ---------- Inference ----------
def predict(event: dict) -> dict:
    """
    Accepts:
      {"pairs": [[user_idx, item_idx], ...]}  OR
      {"rows": [{"mask_id":..., "iid":...} OR {"mask_id":..., "event_description":...}, ...]}
    Returns:
      {"ok": True, "n_rows": N, "predictions": [...], "warnings": [...], "stats": {...}}
    """
    event = event or {}
    # Sanity: confirm we have factors
    uf = getattr(model, "user_factors", None)
    itf = getattr(model, "item_factors", None)
    if uf is None or itf is None:
        logger.error("Model missing factors; cannot score BPR. Type(model)=%s", type(model))
        raise ValueError("Model missing user_factors/item_factors (not an implicit BPR model?).")

    # --- Input logging ---
    keys = list(event.keys())
    logger.info("predict() called with keys=%s", keys)

    rows = event.get("rows") or []
    if "pairs" in event:
        logger.info("Using 'pairs' path: count=%d", len(event["pairs"]))
        logger.debug("pairs preview (first 5): %s", event["pairs"][:5])
    else:
        logger.info("Using 'rows' path: rows_in=%d", len(rows))
        if rows:
            # safe, small preview
            preview = [{k: r.get(k) for k in ("mask_id", "iid", "event_description")} for r in rows[:3]]
            logger.debug("rows preview (first 3): %s", preview)

    pairs, preds = [], []
    stats = {"rows_in": len(rows), "pairs_built": 0,
             "user_miss": 0, "item_miss": 0, "iid_eq_maskid": 0}
    warnings = []

    # --- Build pairs ---
    if "pairs" in event:
        for u, i in event["pairs"]:
            pairs.append((int(u), int(i)))
    else:
        # Log mapping results for first few rows to help debugging
        MAP_LOG_LIMIT = 10
        for idx, r in enumerate(rows):
            u_idx = _user_idx(r.get("mask_id"))
            i_idx = _item_idx(r)

            if str(r.get("iid")) == str(r.get("mask_id")) and r.get("iid") is not None:
                stats["iid_eq_maskid"] += 1

            if u_idx is None:
                stats["user_miss"] += 1
                if len(warnings) < 5:
                    warnings.append(f"row[{idx}] user unmapped: mask_id={r.get('mask_id')}")
                if idx < MAP_LOG_LIMIT:
                    logger.debug("map row[%d]: mask_id=%s -> user_idx=%s", idx, r.get("mask_id"), u_idx)
                continue

            if i_idx is None:
                stats["item_miss"] += 1
                if len(warnings) < 5:
                    warnings.append(f"row[{idx}] item unmapped: iid={r.get('iid')} desc={r.get('event_description')!r}")
                if idx < MAP_LOG_LIMIT:
                    logger.debug("map row[%d]: item from (iid=%s, desc=%r) -> item_idx=%s",
                                 idx, r.get("iid"), r.get("event_description"), i_idx)
                continue

            if idx < MAP_LOG_LIMIT:
                logger.debug("map row[%d]: mask_id=%s -> %s; item -> %s",
                             idx, r.get("mask_id"), u_idx, i_idx)
            pairs.append((u_idx, i_idx))

    stats["pairs_built"] = len(pairs)
    logger.info("Pairs built: %d (user_miss=%d, item_miss=%d, iid_eq_maskid=%d)",
                stats["pairs_built"], stats["user_miss"], stats["item_miss"], stats["iid_eq_maskid"])
    if pairs:
        logger.debug("pairs preview (first 5): %s", pairs[:5])

    # --- Score ---
    for (u, i) in pairs:
        if u < 0 or i < 0 or u >= uf.shape[0] or i >= itf.shape[0]:
            preds.append(0.5)
            if len(warnings) < 5:
                warnings.append(f"out-of-range (u={u}, i={i})")
            logger.debug("out-of-range pair skipped -> 0.5 (u=%s, i=%s)", u, i)
        else:
            score = float(np.dot(uf[u], itf[i]))
            preds.append(_sigmoid(score))

    if preds:
        # Log small sample & summary stats
        arr = np.array(preds, dtype=float)
        logger.info("Predictions: n=%d, min=%.4f, max=%.4f, mean=%.4f", arr.size, float(arr.min()), float(arr.max()), float(arr.mean()))
        logger.debug("Predictions sample (first 10): %s", preds[:10])
    else:
        logger.warning("No predictions produced. Check mapping stats: %s", stats)

    return {"ok": True, "n_rows": len(preds), "predictions": preds, "warnings": warnings, "stats": stats}
