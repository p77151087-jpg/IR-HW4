"""Prepare the locked corpus while building the Community Cloud dependency."""

from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildCorpus(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from ir_hw4 import dataset
        from ir_hw4.io import read_json, sha256

        reference_path = root / "resources" / "reuters-21578-manifest.json"
        reference = read_json(reference_path)
        model = read_json(root / "artifacts" / "reuters-modapte-v1" / "manifest.json")
        if sha256(reference_path) != model["dataset_manifest_sha256"]:
            raise ValueError("The locked Reuters manifest does not match the saved model")

        with TemporaryDirectory(prefix="ir-hw4-reuters-") as directory:
            dataset.DATA = Path(directory)
            for filename, url, digest in (
                ("reuters21578.tar.gz", dataset.DATA_URL, reference["archive_sha256"]),
                ("README.original.txt", dataset.README_URL, reference["readme_sha256"]),
            ):
                path = dataset.DATA / filename
                dataset.download(url, path)
                if sha256(path) != digest:
                    raise ValueError(f"Reuters source SHA256 mismatch: {filename}")

            prepared = dataset.prepare_data()
            prepared.pop("downloaded_at_utc")
            reference.pop("downloaded_at_utc")
            if prepared != reference:
                raise ValueError("The prepared Reuters corpus does not match the saved model inputs")

            package = Path(self.build_lib) / "ir_hw4_cloud_corpus"
            shutil.copyfile(dataset.DATA / "documents.jsonl", package / "documents.jsonl")
            shutil.copyfile(dataset.DATA / "README.original.txt", package / "README.original.txt")
            shutil.copyfile(reference_path, package / "manifest.json")


setup(cmdclass={"build_py": BuildCorpus})
