"""Streamlit Community Cloud entry point for the verified, read-only lab."""

import os
from pathlib import Path

import ir_hw4_cloud_corpus

from ir_hw4 import config


def main():
    os.environ["IR_HW4_CLOUD_READ_ONLY"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    config.DATA = Path(ir_hw4_cloud_corpus.__file__).resolve().parent

    # The UI imports the corpus path and numerical libraries at module load.
    from app import main as render_app

    render_app()


if __name__ == "__main__":
    main()
