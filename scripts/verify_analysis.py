import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ir_hw4.analysis import verify_analysis
from ir_hw4.config import ROOT
from ir_hw4.io import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
args = parser.parse_args()
result = verify_analysis()
if args.output:
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Verification output must stay in HW4")
    if output.exists():
        raise FileExistsError("Choose a new verification output path")
    write_json(output, result)
print(json.dumps(result, ensure_ascii=False, indent=2))
