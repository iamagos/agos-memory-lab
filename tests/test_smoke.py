import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_smoke_receipt_is_exact_and_self_verifying() -> None:
  root = Path(__file__).parents[1]
  completed = subprocess.run(
    [sys.executable, "smoke.py"],
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
  )
  receipt = json.loads(completed.stdout)
  digest = receipt.pop("sha256")

  assert receipt["kernel"] == "0.1.0"
  assert receipt["support"] == {"old": "replaced", "current": "current"}
  assert receipt["selection"]["content"] == "Debt matures in 2029."
  assert receipt["selection"]["source_count"] == 1
  assert receipt["selection"]["included_count"] == 1
  assert receipt["selection"]["truncated"] is False
  assert receipt["selection"]["selected"][0]["source_id"] == "debt-v2"
  assert receipt["selection"]["selected"][0]["paths"] == [
    {"lane": "lexical", "rank": None, "signal": "matched:debt", "relation": None},
    {"lane": "source", "rank": 1, "signal": "exact-source", "relation": None},
  ]
  assert digest == hashlib.sha256(
    json.dumps(receipt, separators=(",", ":"), sort_keys=True).encode()
  ).hexdigest()
