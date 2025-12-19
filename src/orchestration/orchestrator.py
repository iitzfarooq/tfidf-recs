"""
Main orchestrator for ML pipeline workflows.
Manages execution of steps and artifact versioning.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from utils.artifacts_registry import ArtifactsRegistry
from utils.config_loader import ConfigLoader
from orchestration.steps import OrchestrationStep


class Orchestrator:
    """
    Central orchestrator for ML pipeline execution.

    Manages workflow execution, artifact versioning, and step coordination.
    """

    def __init__(
        self,
        steps: List[OrchestrationStep],
        registry: Optional[ArtifactsRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize orchestrator.
        """
        self.steps = steps
        self.config = config or {}

        self.registry = registry
        self.context: Dict[str, Any] = {}

    def run(
        self,
        input_data: Any = None,
        mode: str = "create",
        version_id: Optional[str] = None,
    ) -> str:
        """
        Execute the orchestration pipeline.
        Modes: 'create' (new version), 'load' (existing version).
        Returns the version ID used.
        """
        print("-" * 70)
        print(f"ORCHESTRATOR: Starting pipeline (mode={mode})")
        print("-" * 70)

        # Set up version context
        if mode == "create":
            version_id = self._run_create(input_data=input_data)
        elif mode == "load":
            version_id = self._run_load(input_data=input_data, version_id=version_id)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"✓ Changes committed. Version: {version_id}")
        print("-" * 70)
        return version_id
    
    def _run_create(self, input_data: Any = None) -> str:
        with self.registry(mode='create') as reg:
            self.context['registry'] = reg
            self.context['version_id'] = reg.active_version
            
            print(f"✓ Created new version: {reg.active_version}\n")
            
            data = input_data
            for i, step in enumerate(self.steps, 1):
                print(f"Step {i}/{len(self.steps)}: {step.name}")
                data = step.execute(data, self.context)
                print()
            
            version_id = reg.active_version

        return version_id

    def _run_load(self, input_data: Any = None, version_id: Optional[str] = None) -> str:
        with self.registry(mode='load', version_id=version_id) as reg:
            self.context['registry'] = reg
            self.context['version_id'] = reg.active_version
            
            print(f"✓ Loaded version: {reg.active_version}\n")
            
            data = input_data
            for i, step in enumerate(self.steps, 1):
                print(f"Step {i}/{len(self.steps)}: {step.name}")
                data = step.execute(data, self.context)
                print()
            
            version_id = reg.active_version

        return version_id

    def run_steps(
        self,
        step_names: List[str],
        input_data: Any = None,
        version_id: Optional[str] = None,
    ) -> str:
        """
        Execute only specific steps from the pipeline.

        Args:
            step_names: Names of steps to execute
            input_data: Initial input data
            version_id: Optional version to work with

        Returns:
            Version ID used
        """
        # Filter steps
        filtered_steps = [step for step in self.steps if step.name in step_names]

        if not filtered_steps:
            raise ValueError(f"No matching steps found for: {step_names}")

        print(f"Running {len(filtered_steps)} step(s): {step_names}")

        with self.registry(mode='load', version_id=version_id) as reg:
            self.context["registry"] = reg
            self.context["version_id"] = reg.active_version
            
            data = input_data
            for step in filtered_steps:
                print(f"\nExecuting: {step.name}")
                data = step.execute(data, self.context)

        return version_id


def create_orchestrator(
    config: Optional[Dict[str, Any]] = None, registry=None
) -> Orchestrator:
    """
    Factory function to create an orchestrator from configuration.

    Args:
        config: Configuration dictionary with orchestration settings

    Returns:
        Configured Orchestrator instance
    """
    from orchestration.steps import (
        LoadDataStep,
        FitVectorizerStep,
        GenerateFeaturesStep,
        GenerateSimilarityStep,
    )

    config = config or {}

    # Build steps from config
    steps = []
    step_config = config.get("steps", {})

    steps.append(LoadDataStep(step_config.get("load_data", {})))
    steps.append(FitVectorizerStep(step_config.get("vectorizer", {})))
    steps.append(GenerateFeaturesStep(step_config.get("features", {})))
    steps.append(GenerateSimilarityStep(step_config.get("similarity", {})))

    return Orchestrator(steps=steps, config=config, registry=registry)
