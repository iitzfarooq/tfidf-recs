import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class ArtifactsManager:
    """
    Manages saving and versioning of artifacts such as matrices, and recommendations.
    """
    
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.registry_path = self.artifacts_dir / "artifacts_registry.json"
        self.registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self):
        if not self.registry_path.exists():
            self.registry = {}
            return
        
        with open(self.registry_path, 'r') as f:
            self.registry = json.load(f)

    def _save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)

    def save_artifact(
        self, 
        artifact_type: str,
        artifact_name: str,
        data: Any,
        metadata: Dict[str, Any],
        ext: str = 'json',
        version: str = None
    ):
        if version is None:
            version = f"1.0.0.{datetime.now().strftime('%Y%m%d%H%M%S')}"

        version_dir = self.artifacts_dir / version / artifact_type
        version_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = version_dir / f"{artifact_name}.{ext}"

        if ext == 'json':
            with open(artifact_path, 'w') as f:
                json.dump(data, f)
        elif ext == 'npz':
            import scipy.sparse as sparse
            sparse.save_npz(artifact_path, data)

        self.registry[artifact_type][artifact_name] = {
            'version': version,
            'path': str(artifact_path),
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }

        self._save_registry()