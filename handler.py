# olg-reco-container/handler.py
import json, os, time, uuid, logging
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

from recommender import predict

# ---- Env vars (support a few names) ----
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET") or os.getenv("S3_BUCKET")
_raw_prefix = os.getenv("OUTPUT_PREFIX") or os.getenv("OUT_PREFIX") or os.getenv("S3_PREFIX") or "outputs"
OUTPUT_PREFIX = _raw_prefix.strip().strip("/")  # normalize: no leading/trailing slash

s3 = boto3.client("s3")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def put_json_s3(bucket: str, prefix: str, payload: dict, key_hint: str = "result") -> str:
    """Upload JSON to s3://bucket/prefix/key_hint__YYYYMMDDTHHMMSSZ.json"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"{prefix}/{key_hint}__{ts}.json" if prefix else f"{key_hint}__{ts}.json"
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    try:
        resp = s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        logger.info("S3 put_object ok: %s %s", bucket, key)
        return f"s3://{bucket}/{key}"
    except ClientError as e:
        logger.error("S3 put_object failed: %s", e, exc_info=True)
        raise

def lambda_handler(event, context):
    req_id = getattr(context, "aws_request_id", str(uuid.uuid4()))
    start = time.time()

    # 1) run model
    try:
        model_out = predict(event or {})
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"ok": False, "error": str(e), "request_id": req_id}),
            "headers": {"Content-Type": "application/json"},
        }

    # 2) assemble payload
    result = {
        **model_out,
        "request_id": req_id,
        "latency_ms": round((time.time() - start) * 1000, 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # 3) write to S3 if configured
    if not OUTPUT_BUCKET:
        result.setdefault("warnings", []).append("OUTPUT_BUCKET not set; skipping S3 write")
        return {"statusCode": 200, "body": json.dumps(result), "headers": {"Content-Type": "application/json"}}

    try:
        uri = put_json_s3(OUTPUT_BUCKET, OUTPUT_PREFIX, result, key_hint="daily_scores")
        result["s3_uri"] = uri
    except Exception as e:
        result.setdefault("warnings", []).append(f"S3 write failed: {e}")

    return {"statusCode": 200, "body": json.dumps(result), "headers": {"Content-Type": "application/json"}}
