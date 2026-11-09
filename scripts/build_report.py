import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ir_hw4.analysis import build_report

parser = argparse.ArgumentParser(description="Build a new evidence-backed report; preserve all models and original experiments")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
print(build_report(args.output))
