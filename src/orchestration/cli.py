"""
CLI interface for orchestration workflows using Click.
"""

import sys
import click
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.artifacts_registry import ArtifactsRegistry
from src.orchestration.orchestrator import create_orchestrator


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Orchestration CLI for ML pipelines with versioned artifacts."""
    pass


@cli.command()
@click.option(
    '--input', 
    type=click.Path(exists=True),
    help='Input data file path'
)
@click.option(
    '--steps',
    type=str,
    help='Comma-separated list of step names to run'
)
@click.option(
    '--version',
    type=str,
    help='Version ID to use (for partial runs)'
)
@click.option(
    '--data-config',
    is_flag=True,
    help='Use input path from data config'
)
def run(input, steps, version, data_config):
    """Execute the full pipeline or specific steps."""
    # Load configurations
    config_loader = ConfigLoader()
    config_loader.load_all()
    
    orchestration_config = config_loader.get('orchestration', {})
    data_config_dict = config_loader.get('data', {})
    
    # Get input path
    if input:
        input_path = input
    elif data_config:
        input_path = data_config_dict.get('output_path')
    else:
        input_path = orchestration_config.get('steps', {}).get(
            'load_data', {}
        ).get('input_path')
    
    if not input_path:
        click.echo(click.style("Error: No input path specified", fg='red'))
        click.echo("Use --input or configure in orchestration_config.yaml")
        sys.exit(1)
    
    orchestrator = create_orchestrator(orchestration_config)
    
    # Execute
    try:
        if steps:
            # Run specific steps
            step_names = [s.strip() for s in steps.split(',')]
            version_id = orchestrator.run_steps(
                step_names=step_names,
                input_data=input_path,
                version_id=version
            )
        else:
            # Run full pipeline
            version_id = orchestrator.run(
                input_data=input_path,
                mode='create'
            )
        
        click.echo()
        click.echo(click.style("✓ Pipeline completed successfully!", fg='green'))
        click.echo(f"Version: {click.style(version_id, fg='cyan')}")
    except Exception as e:
        click.echo(click.style(f"✗ Pipeline failed: {e}", fg='red'))
        sys.exit(1)


@cli.command()
@click.option(
    '--version',
    type=str,
    help='Version ID to load (default: latest)'
)
def load(version):
    """Load and inspect a version."""
    config_loader = ConfigLoader()
    config_loader.load_all()
    registry_config = config_loader.get('registry')
    
    registry = ArtifactsRegistry(registry_config)
    
    try:
        if version:
            registry.load_version(version)
            version_id = version
        else:
            registry.load_latest()
            version_id = registry.active_version
        
        click.echo(f"Loaded version: {click.style(version_id, fg='cyan')}\n")
        click.echo("Artifacts:")
        
        version_obj = registry.versions[version_id]
        for artifact_type, artifacts in version_obj.artifacts.items():
            click.echo(f"  {click.style(artifact_type, fg='yellow')}:")
            for name, meta in artifacts.items():
                click.echo(f"    - {name} ({meta.path.name})")
        
        # Show metadata if available
        if 'metadata' in version_obj.artifacts:
            try:
                metadata = registry.get_artifact('metadata', 'metadata')
                click.echo(f"\n{click.style('Metadata:', fg='yellow')}")
                for key, value in metadata.items():
                    click.echo(f"  {key}: {value}")
            except:
                pass
    
    except FileNotFoundError as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'))
        sys.exit(1)


@cli.command('list-versions')       # hyphen separator for CLI
def list_versions():
    """List all available versions."""
    config_loader = ConfigLoader()
    config_loader.load_all()
    registry_config = config_loader.get('registry')
    
    registry = ArtifactsRegistry(registry_config)
    versions = registry.list_versions()
    
    if not versions:
        click.echo("No versions found")
        return
    
    click.echo(f"Found {click.style(str(len(versions)), fg='cyan')} version(s):\n")
    
    for ver in versions:
        registry.load_version(ver)
        
        artifact_count = sum(
            len(arts) 
            for arts in registry.versions[ver].artifacts.values()
        )
        
        click.echo(f"  {click.style(ver, fg='cyan')}")
        click.echo(f"    Artifacts: {artifact_count}")
        
        click.echo()


@cli.command()
@click.option(
    '--type',
    'config_type',
    type=str,
    help='Config type to show (orchestration, registry, etc.)'
)
def config(config_type):
    """Display current configuration."""
    config_loader = ConfigLoader()
    config_loader.load_all()
    
    click.echo(click.style("Current Configuration:", fg='yellow', bold=True))
    click.echo()
    
    if config_type:
        cfg = config_loader.get(config_type)
        click.echo(f"{click.style(config_type.upper() + ' Config:', fg='cyan')}")
        click.echo(cfg)
    else:
        # Show all configs
        for name, cfg in config_loader.configs.items():
            click.echo(f"{click.style(name.upper() + ' Config:', fg='cyan')}")
            click.echo(cfg)
            click.echo()


if __name__ == '__main__':
    cli()
