"""Run existing verification without overwriting historical verification reports."""
import json
from pathlib import Path
import runpy
import sys

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
sys.path.insert(0, str(ROOT))
import ir_hw4.io as project_io

original_write_json = project_io.write_json
redirects = {
    ROOT / "reports/verification.json": AUDIT / "verification.json",
    ROOT / "reports/hw3-protection-after.json": AUDIT / "hw3-protection-after.json",
}

def audit_write_json(path, value):
    resolved = Path(path).resolve()
    if resolved not in redirects:
        raise RuntimeError(f"Unexpected verification write: {resolved}")
    return original_write_json(redirects[resolved], value)

project_io.write_json = audit_write_json
sys.argv = [str(ROOT / "scripts/verify_hw4.py")]
runpy.run_path(sys.argv[0], run_name="__main__")
project_io.write_json = original_write_json

from ir_hw4.analysis import verify_analysis
analysis = verify_analysis()
original_write_json(AUDIT / "analysis-verification.json", analysis)
print(json.dumps(analysis, ensure_ascii=False, indent=2))
