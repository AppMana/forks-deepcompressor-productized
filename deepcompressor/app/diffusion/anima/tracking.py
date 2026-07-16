"""MLflow experiment tracking for the Anima SVDQuant workflow."""

from __future__ import annotations

import json
import os
import subprocess
import typing as tp
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_TRACKING_URI = "https://mlflow.appmana.com"
DEFAULT_EXPERIMENT_NAME = "anima-aesthetic-v1.1-svdquant"
REFERENCE_FILENAME = "mlflow-run.json"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _clean_value(value: tp.Any) -> str | float | int | bool:
    if value is None:
        return ""
    if isinstance(value, (str, float, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)


@dataclass(frozen=True)
class RunReference:
    """The non-secret information needed to resume an MLflow run."""

    tracking_uri: str
    experiment_name: str
    experiment_id: str
    run_id: str
    run_name: str

    @classmethod
    def load(cls, path: str | Path) -> RunReference:
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path


class ExperimentTracker:
    """Small MLflow facade that remains a no-op when tracking is disabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        tracking_uri: str = "",
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        run_name: str = "",
        run_id: str = "",
        tags: dict[str, tp.Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        self.experiment_name = experiment_name
        self.requested_run_name = run_name
        self.requested_run_id = run_id
        self.tags = tags or {}
        self.reference: RunReference | None = None
        self._mlflow = None

    def __enter__(self) -> ExperimentTracker:
        if not self.enabled:
            return self
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        tags = {
            "project": "deepcompressor-svdquant",
            "model": "anima-aesthetic-v1.1",
            "git.commit": _git_commit(),
            **{key: str(_clean_value(value)) for key, value in self.tags.items()},
        }
        if self.requested_run_id:
            run = mlflow.start_run(run_id=self.requested_run_id, tags=tags)
            experiment = mlflow.get_experiment(run.info.experiment_id)
        else:
            experiment = mlflow.set_experiment(self.experiment_name)
            run = mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=self.requested_run_name or None,
                tags=tags,
            )
        self.reference = RunReference(
            tracking_uri=self.tracking_uri,
            experiment_name=experiment.name,
            experiment_id=run.info.experiment_id,
            run_id=run.info.run_id,
            run_name=run.data.tags.get("mlflow.runName", self.requested_run_name),
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._mlflow is not None:
            self._mlflow.end_run(status="FAILED" if exc_type is not None else "FINISHED")
        return False

    @property
    def run_id(self) -> str:
        return self.reference.run_id if self.reference is not None else ""

    def save_reference(self, path: str | Path) -> Path | None:
        if self.reference is None:
            return None
        return self.reference.save(path)

    def log_params(self, params: dict[str, tp.Any]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params({key: _clean_value(value) for key, value in params.items()})

    def log_metrics(self, metrics: dict[str, float | int], *, step: int | None = None) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)

    def set_tags(self, tags: dict[str, tp.Any]) -> None:
        if self._mlflow is not None:
            self._mlflow.set_tags({key: str(_clean_value(value)) for key, value in tags.items()})

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        path = Path(path)
        if self._mlflow is not None and path.is_file():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, path: str | Path, artifact_path: str | None = None) -> None:
        path = Path(path)
        if self._mlflow is not None and path.is_dir():
            self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def find_run_reference(manifest: str | Path, explicit: str = "") -> Path | None:
    """Find the quantization run reference associated with a manifest."""

    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    manifest = Path(manifest).expanduser().resolve()
    for parent in manifest.parents:
        candidate = parent / REFERENCE_FILENAME
        if candidate.is_file():
            return candidate
    return None
