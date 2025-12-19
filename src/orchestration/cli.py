"""
CLI interface for orchestration workflows using Click.
"""

import sys
import click
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root and src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # For internal imports
sys.path.insert(0, str(PROJECT_ROOT))  # For src.* imports

from src.utils.config_loader import ConfigLoader
from src.utils.artifacts_registry import ArtifactsRegistry
from src.orchestration.orchestrator import create_orchestrator


def get_config_loader() -> ConfigLoader:
    """Helper to get initialized config loader."""
    loader = ConfigLoader(config_dir=str(PROJECT_ROOT / "configs"))
    loader.load_all()
    return loader


def get_registry_config(config_loader: ConfigLoader) -> Dict[str, Any]:
    """Helper function to load registry config with absolute paths."""
    registry_config = config_loader.get("registry")

    if not registry_config:
        click.echo(click.style("✗ Error: Registry configuration not found", fg="red"))
        sys.exit(1)

    # Resolve base_path to absolute path
    if "base_path" in registry_config:
        registry_config["base_path"] = str(PROJECT_ROOT / registry_config["base_path"])

    return registry_config


def get_registry(config_loader: Optional[ConfigLoader] = None) -> ArtifactsRegistry:
    """Helper to get initialized registry."""
    if config_loader is None:
        config_loader = get_config_loader()
    config = get_registry_config(config_loader)
    return ArtifactsRegistry(config)


def get_orchestrator(config_loader: ConfigLoader, registry: ArtifactsRegistry):
    """Helper to get initialized orchestrator."""
    orchestration_config = config_loader.get("orchestration", {})
    orchestrator = create_orchestrator(orchestration_config, registry=registry)
    return orchestrator


def resolve_input_path(
    input_arg: Optional[str], data_config_flag: bool, config_loader: ConfigLoader
) -> str:
    """Resolve the input data path from arguments or config."""
    orchestration_config = config_loader.get("orchestration", {})
    data_config_dict = config_loader.get("data", {})

    if input_arg:
        input_path = input_arg
    elif data_config_flag:
        input_path = data_config_dict.get("output_path")
    else:
        input_path = (
            orchestration_config.get("steps", {}).get("load_data", {}).get("input_path")
        )

    if not input_path:
        click.echo(click.style("Error: No input path specified", fg="red"))
        click.echo("Use --input or configure in orchestration_config.yaml")
        sys.exit(1)

    # Resolve input path to absolute if relative
    if not Path(input_path).is_absolute():
        input_path = str(PROJECT_ROOT / input_path)

    return input_path


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Orchestration CLI for ML pipelines with versioned artifacts."""
    pass


@cli.command()
@click.option("--input", type=click.Path(exists=True), help="Input data file path")
@click.option("--steps", type=str, help="Comma-separated list of step names to run")
@click.option("--version", type=str, help="Version ID to use (for partial runs)")
@click.option("--data-config", is_flag=True, help="Use input path from data config")
def run(input, steps, version, data_config):
    """Execute the full pipeline or specific steps."""
    config_loader = get_config_loader()
    input_path = resolve_input_path(input, data_config, config_loader)

    # Initialize components
    registry = get_registry(config_loader)
    orchestrator = get_orchestrator(config_loader, registry)

    try:
        if steps:
            # Run specific steps on an existing version
            version_id = orchestrator.run_steps(
                step_names=[s.strip() for s in steps.split(",")],
                input_data=input_path,
                version_id=version,
            )
        elif version:
            # Load existing version
            version_id = orchestrator.run(
                input_data=input_path, mode="load", version_id=version
            )
        else:
            # Create new version
            version_id = orchestrator.run(input_data=input_path, mode="create")

        click.echo(click.style("✓ Pipeline completed successfully!", fg="green"))
        click.echo(f"Version: {click.style(version_id, fg='cyan')}")

    except Exception as e:
        click.echo(click.style(f"✗ Pipeline failed: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option("--version", type=str, help="Version ID to load (default: latest)")
def load(version):
    """Load and inspect a version."""
    registry = get_registry()

    try:
        if version:
            registry.load_version(version)
        else:
            registry.load_latest()
            
        version_id = registry.active_version
        click.echo(f"Loaded version: {click.style(version_id, fg='cyan')}\n")

        version_obj = registry.versions[version_id]
        _print_artifacts(version_obj, registry)

    except FileNotFoundError as e:
        click.echo(click.style(f"✗ Error: {e}", fg="red"))
        sys.exit(1)


def _print_artifacts(version_obj, registry):
    """Helper to print artifacts for a version."""
    click.echo("Artifacts:")
    for artifact_type, artifacts in version_obj.artifacts.items():
        click.echo(f"  {click.style(artifact_type, fg='yellow')}:")
        for name, meta in artifacts.items():
            click.echo(f"    - {name} ({meta.path.name})")

    # Show metadata if available
    if "metadata" in version_obj.artifacts:
        try:
            metadata = registry.get_artifact("metadata", "metadata")
            click.echo(f"\n{click.style('Metadata:', fg='yellow')}")
            for key, value in metadata.items():
                if key != "movie_ids":
                    click.echo(f"  {key}: {value}")
        except Exception:
            pass


@cli.command("list-versions")
def list_versions():
    """List all available versions."""
    registry = get_registry()
    versions = registry.list_versions()

    if not versions:
        click.echo("No versions found")
        return

    click.echo(f"Found {click.style(str(len(versions)), fg='cyan')} version(s):\n")

    for ver in versions:
        try:
            registry.load_version(ver)
            version_obj = registry.versions[ver]
            artifact_count = sum(len(arts) for arts in version_obj.artifacts.values())
            click.echo(f"  {click.style(ver, fg='cyan')} (Artifacts: {artifact_count})")
        except Exception:
            click.echo(f"  {click.style(ver, fg='red')} (Error loading)")

    click.echo()


@cli.command()
@click.option("--type", "config_type", type=str, help="Config type to show")
def config(config_type):
    """Display current configuration."""
    config_loader = get_config_loader()

    click.echo(click.style("Current Configuration:", fg="yellow", bold=True))
    click.echo()

    if config_type:
        cfg = config_loader.get(config_type)
        click.echo(f"{click.style(config_type.upper() + ' Config:', fg='cyan')}")
        click.echo(cfg)
    else:
        for name, cfg in config_loader.configs.items():
            click.echo(f"{click.style(name.upper() + ' Config:', fg='cyan')}")
            click.echo(cfg)
            click.echo()


if __name__ == "__main__":
    cli()
