import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ir_hw4.config import ROOT, RUN
from ir_hw4.io import write_json
from ir_hw4.validation import verify, verify_hw3

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=RUN)
args = parser.parse_args()
result = verify(args.run)
write_json(ROOT / "reports/verification.json", result)
protection = verify_hw3()
write_json(ROOT / "reports/hw3-protection-after.json", protection)
print(result)
print(protection)
