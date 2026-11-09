"""Launch a local training process and publish its latest progress to the site."""

from contextlib import contextmanager
from datetime import datetime, timezone
import os
import subprocess
import sys
from threading import Lock
from time import sleep, time
from uuid import uuid4

from .config import ROOT
from .experiments import experiment_paths
from .io import read_json, write_json
from .parameter_training import model_configs, train_parameter_experiment

JOBS = ROOT / "reports" / "training-jobs"
ACTIVE_PHASES = {"queued", "preparing", "training", "saving", "evaluating"}
MIN_WEB_VECTOR_SIZE = 100


def job_directory(name):
    experiment_paths(name)
    directory = JOBS / name
    if directory.resolve() != JOBS.resolve() / name or not directory.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("訓練工作目錄超出專案範圍。")
    return directory


@contextmanager
def worker_slot():
    with (JOBS / "active.lock").open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("網站已有一組訓練進行中。") from error
        yield


def write_status(directory, phase, **details):
    marker = directory / "status.json"
    pending = directory / "status.json.tmp"
    write_json(pending, {"name": directory.name, "phase": phase, **details})
    for attempt in range(5):
        try:
            pending.replace(marker)
            return
        except PermissionError:
            if attempt == 4:
                raise
            sleep(.025)


class TrainingJobs:
    def __init__(self):
        self.lock = Lock()
        self.directory = None
        self.process = None

    def start(self, name, settings):
        configs = model_configs(settings)
        if any(config.vector_size < MIN_WEB_VECTOR_SIZE for config in configs.values()):
            raise ValueError(f"網站訓練的向量維度至少為 {MIN_WEB_VECTOR_SIZE}。")
        name = name.strip() or f"web-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
        paths = experiment_paths(name)
        directory = job_directory(name)
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("網站已有一組訓練進行中。")
            if directory.exists() or any(path.exists() for path in paths):
                raise FileExistsError(f"實驗名稱 {name} 已使用，請輸入新名稱。")
            JOBS.mkdir(parents=True, exist_ok=True)
            with worker_slot():
                directory.mkdir()
                write_json(directory / "request.json", settings)
                write_status(directory, "queued")
            self.directory, self.process = directory, None
            try:
                with (directory / "worker.log").open("wb") as log:
                    self.process = subprocess.Popen(
                        [sys.executable, "-B", "-X", "utf8", "-m", "ir_hw4.training_jobs", name],
                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
            except OSError as error:
                write_status(directory, "failed", error=str(error))
                raise
        return name

    def latest(self):
        with self.lock:
            if self.directory is None and JOBS.is_dir():
                self.directory = max(
                    (path for path in JOBS.iterdir() if (path / "status.json").is_file()),
                    key=lambda path: (path / "status.json").stat().st_mtime, default=None,
                )
            if self.directory is None:
                return None
            marker = self.directory / "status.json"
            status = read_json(marker)
            if status["phase"] not in ACTIVE_PHASES:
                return status
            if self.process is not None:
                if self.process.poll() is None:
                    return status
            else:
                if status["phase"] == "queued" and time() - marker.stat().st_mtime < 45:
                    return status
                try:
                    with worker_slot():
                        status = read_json(marker)
                except RuntimeError:
                    return status
            status = read_json(marker)
            if status["phase"] in ACTIVE_PHASES:
                return {**status, "phase": "failed", "error": "訓練程序提前結束，請查看該次訓練記錄。"}
            return status


def run_job(name):
    directory = job_directory(name)
    try:
        with worker_slot():
            train_parameter_experiment(
                name, read_json(directory / "request.json"),
                lambda phase, **details: write_status(directory, phase, **details),
            )
            write_status(directory, "complete")
    except Exception as error:
        write_status(directory, "failed", error=str(error))
        raise


if __name__ == "__main__":
    run_job(sys.argv[1])
