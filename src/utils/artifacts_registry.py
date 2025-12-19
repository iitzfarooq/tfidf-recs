from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from utils.load_save import *


@dataclass
class ArtifactMetadata:
    name: str
    path: Path
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactVersion:
    version_id: str
    timestamp: str
    artifacts: Dict[str, Dict[str, ArtifactMetadata]] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ArtifactVersion":
        artifacts = {
            type: {
                name: ArtifactMetadata(
                    name=meta["name"],
                    path=Path(meta["path"]),
                    created_at=meta["created_at"],
                    metadata=meta.get("metadata", {}),
                )
                for name, meta in type_dict.items()
            }
            for type, type_dict in data.get("artifacts", {}).items()
        }
        return ArtifactVersion(
            version_id=data.get("version_id"),
            timestamp=data.get("timestamp"),
            artifacts=artifacts,
        )


class VersionContext:
    def __init__(self, registry: "ArtifactsRegistry", mode: str = "load"):
        self.registry = registry
        self.mode = mode
        self.previous_version = registry.active_version

    def __enter__(self):
        if self.mode == "create":
            self.registry.create_version()
        elif self.mode == "load":
            self.registry.load_latest()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return self.registry

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None and self.mode == "create":
            self.registry.commit()
        else:
            self.registry.active_version = self.previous_version

        return False  # Do not suppress exceptions


@dataclass
class ArtifactConfig:
    format: str
    compression: Optional[str] = None
    required: bool = True
    dependencies: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ArtifactConfig":
        return ArtifactConfig(
            format=data.get("format"),
            compression=data.get("compression"),
            required=data.get("required", True),
            dependencies=data.get("dependencies", []),
        )


class ArtifactsRegistry:
    """
    Centralized registry for managing artifacts with versioning,
    dependencies, and required artifacts.

    ArtifactsRegistry(
        base_path, base_version, artifact_types: {str -> ArtifactConfig}
    )

    Operations:
    - load_latest()
    - load_version(version_id)
    - list_versions()
    - create_version()
    - register_artifact(artifact_type, artifact_name, artifact_data, metadata)
    - get_artifact(artifact_type, artifact_name)
    - commit()

    Usage:
    with registry(mode='create') as reg:
    """

    def __init__(self, config: Dict[str, Any]):
        self.base_path = Path(config.get("base_path", "data/artifacts"))
        self.base_version = config.get("base_version", "v1")
        self.artifacts_config = {
            type: ArtifactConfig.from_dict(cfg)
            for type, cfg in config.get("artifact_types", {}).items()
        }

        self.base_path.mkdir(parents=True, exist_ok=True)

        self.versions: Dict[str, ArtifactVersion] = {}
        self.active_version: Optional[str] = None
        self.pending_artifacts: Dict[str, Any] = {}

    def load_latest(self) -> None:
        """
        Load the latest version of artifacts from the registry.
        Sets active_version accordingly.
        """

        candidates = self._find_version_candidates()

        if not candidates:
            raise FileNotFoundError("No versions found in the registry.")

        latest_version = self._get_latest_version(candidates)
        self.active_version = latest_version

        version_data = self._load_version_metadata(latest_version)
        self.versions[latest_version] = ArtifactVersion.from_dict(version_data)

    def load_version(self, version_id: str) -> None:
        """
        Load a specific version by its version ID. Raises FileNotFoundError
        if the version does not exist.
        """
        self.active_version = version_id
        version_data = self._load_version_metadata(version_id)
        self.versions[version_id] = ArtifactVersion.from_dict(version_data)

    def list_versions(self) -> List[str]:
        """
        List all available versions in the registry. Returns a sorted list
        of version IDs in descending order.
        """
        candidates = self._find_version_candidates()
        return sorted(candidates, reverse=True)

    def _find_version_candidates(self) -> list:
        pattern = re.compile(rf"^{self.base_version}\.(\d{{8}}_\d{{6}})$")
        return [
            entry.name
            for entry in self.base_path.iterdir()
            if entry.is_dir() and pattern.match(entry.name)
        ]

    def _get_latest_version(self, candidates: list) -> str:
        return max(candidates)

    def _load_version_metadata(self, version: str) -> dict:
        version_path = self.base_path / version
        version_file = version_path / "version_metadata.json"
        if not version_file.exists():
            raise FileNotFoundError(
                f"Version metadata file not found for version '{version}'."
            )
        return load_json(version_file)

    def create_version(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{self.base_version}.{timestamp}"

        self.versions[version_id] = ArtifactVersion(
            version_id=version_id, timestamp=timestamp, artifacts={}
        )
        self.active_version = version_id

    def register_artifact(
        self,
        artifact_type: str,
        artifact_name: str,
        artifact_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ArtifactsRegistry":
        if self.active_version is None:
            raise RuntimeError("No active version set.")

        if artifact_type not in self.artifacts_config:
            raise ValueError(f"Unknown artifact type: '{artifact_type}'.")

        self._validate_dependencies(artifact_type)

        version = self.versions[self.active_version]
        artifact_path = self.base_path / self.active_version / artifact_type
        format_cfg = self.artifacts_config[artifact_type]

        artifact_file = artifact_path / f"{artifact_name}.{format_cfg.format}"
        artifact_meta = ArtifactMetadata(
            name=artifact_name,
            path=artifact_file,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        version.artifacts.setdefault(artifact_type, {})[artifact_name] = artifact_meta
        self.pending_artifacts[(version.version_id, artifact_type, artifact_name)] = (
            artifact_data
        )

        return self

    def get_artifact(self, artifact_type: str, artifact_name: str) -> Any:
        if self.active_version is None:
            raise RuntimeError("Error: No active version set.")

        version = self.versions[self.active_version]

        self._check_artifact_exists(version, artifact_type, artifact_name)
        return self._load_artifact_data(
            version.artifacts[artifact_type][artifact_name], artifact_type
        )

    def _check_artifact_exists(
        self, version: ArtifactVersion, artifact_type: str, artifact_name: str
    ) -> None:
        if artifact_type not in version.artifacts:
            raise KeyError(
                f"Artifact type '{artifact_type}' not found in "
                f"version '{version.version_id}'."
            )

        if artifact_name not in version.artifacts[artifact_type]:
            raise KeyError(
                f"Artifact '{artifact_name}' of type '{artifact_type}' "
                f"not found in version '{version.version_id}'."
            )

    def _load_artifact_data(
        self, artifact_meta: ArtifactMetadata, artifact_type: str
    ) -> Any:
        """Load artifact data from file, using cache if available."""
        art_key = (self.active_version, artifact_type, artifact_meta.name)

        # Check cache first
        if art_key in self.pending_artifacts:
            return self.pending_artifacts[art_key]

        # Load from file
        if not artifact_meta.path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_meta.path}")

        data = self._load_from_file(artifact_meta.path)

        # Cache the loaded data in pending_artifacts
        self.pending_artifacts[art_key] = data
        return data

    def _load_from_file(self, file_path: Path) -> Any:
        """Load data from file based on format."""
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return load_json(file_path)

        elif suffix == ".yaml" or suffix == ".yml":
            return load_yaml(file_path)

        elif suffix == ".pkl" or suffix == ".pickle":
            return load_pickle(file_path)

        elif suffix == ".joblib":
            return load_joblib(file_path)

        elif suffix == ".npz":
            return load_npz(file_path)

        elif suffix == ".parquet":
            return load_parquet(file_path)

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _save_to_file(self, data: Any, file_path: Path) -> None:
        """Save data to file based on format."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            save_json(data, file_path)

        elif suffix == ".yaml" or suffix == ".yml":
            save_yaml(data, file_path)

        elif suffix == ".pkl" or suffix == ".pickle":
            save_pickle(data, file_path)

        elif suffix == ".joblib":
            save_joblib(data, file_path)

        elif suffix == ".npz":
            save_npz(data, file_path)

        elif suffix == ".parquet":
            save_parquet(data, file_path)

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def commit(self) -> None:
        if self.active_version is None:
            raise RuntimeError("No active version set.")

        if not self.pending_artifacts:
            return

        version = self.versions[self.active_version]

        self._validate_required_artifacts(version)

        # Save each pending artifact
        for key, data in self.pending_artifacts.items():
            version_id, artifact_type, artifact_name = key

            if version_id != self.active_version:
                continue

            artifact_meta = version.artifacts[artifact_type][artifact_name]
            self._save_to_file(data, artifact_meta.path)

        self._save_version_metadata(version)

        # Clear pending artifacts for this version
        self.pending_artifacts = {
            k: v
            for k, v in self.pending_artifacts.items()
            if k[0] != self.active_version
        }

    def _validate_dependencies(self, artifact_type: str) -> None:
        config = self.artifacts_config[artifact_type]
        dependencies = config.dependencies

        if not dependencies:
            return

        version = self.versions[self.active_version]
        missing_deps = []

        for dep_type in dependencies:
            if dep_type not in version.artifacts or not version.artifacts[dep_type]:
                missing_deps.append(dep_type)

        if missing_deps:
            raise ValueError(
                f"Cannot register '{artifact_type}': missing required "
                f"dependencies {missing_deps}"
            )

    def _validate_required_artifacts(self, version: ArtifactVersion) -> None:
        missing_required = []

        for artifact_type, config in self.artifacts_config.items():
            if config.required:
                if (
                    artifact_type not in version.artifacts
                    or not version.artifacts[artifact_type]
                ):
                    missing_required.append(artifact_type)

        if missing_required:
            raise ValueError(
                f"Missing required artifacts in version "
                f"'{version.version_id}': {missing_required}"
            )

    def _save_version_metadata(self, version: ArtifactVersion) -> None:
        version_path = self.base_path / version.version_id
        version_path.mkdir(parents=True, exist_ok=True)

        metadata_file = version_path / "version_metadata.json"

        metadata = {
            "version_id": version.version_id,
            "timestamp": version.timestamp,
            "artifacts": {
                artifact_type: {
                    name: {
                        "name": meta.name,
                        "path": str(meta.path),
                        "created_at": meta.created_at,
                        "metadata": meta.metadata,
                    }
                    for name, meta in artifacts.items()
                }
                for artifact_type, artifacts in version.artifacts.items()
            },
        }

        save_json(metadata, metadata_file)

    def __call__(self, mode: str = "load") -> VersionContext:
        return VersionContext(self, mode)
