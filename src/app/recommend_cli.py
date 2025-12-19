"""
Movie Recommendation CLI - User Interface
Separate from pipeline orchestration.
"""

import sys
import click
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent

from src.utils.artifacts_registry import ArtifactsRegistry
from src.utils.config_loader import ConfigLoader
from src.engine.recommender import ContentBasedRecommender
from src.engine.similarity_strategies import create_similarity_strategy

# --- Helpers ---


def get_registry_config() -> Dict[str, Any]:
    """Load registry config with absolute paths."""
    config_loader = ConfigLoader(config_dir=str(PROJECT_ROOT / "configs"))
    config_loader.load_all()
    registry_config = config_loader.get("registry")

    if not registry_config:
        click.echo(click.style("✗ Error: Registry configuration not found", fg="red"))
        sys.exit(1)

    if "base_path" in registry_config:
        registry_config["base_path"] = str(PROJECT_ROOT / registry_config["base_path"])

    return registry_config


def load_movies() -> pd.DataFrame:
    """Load movies dataset."""
    movies_path = PROJECT_ROOT / "data" / "raw" / "movies.csv"
    if not movies_path.exists():
        click.echo(click.style("✗ Error: movies.csv not found", fg="red"))
        sys.exit(1)
    return pd.read_csv(movies_path)


def load_recommender_system(
    version: Optional[str] = None,
) -> Tuple[ContentBasedRecommender, Dict[str, Any]]:
    """Load recommendation artifacts and create recommender."""
    registry_config = get_registry_config()
    registry = ArtifactsRegistry(registry_config)

    try:
        with registry(mode="load", version_id=version) as reg:
            feature_matrix = reg.get_artifact("feature_matrix", "features")
            metadata = reg.get_artifact("metadata", "metadata")

            movie_ids = metadata.get("movie_ids", [])
            strategy = create_similarity_strategy("cosine")

            recommender = ContentBasedRecommender(
                item_ids=movie_ids,
                features=feature_matrix,
                similarity_strategy=strategy,
            )

            return recommender, metadata
    except FileNotFoundError:
        click.echo(click.style("✗ Error: No recommendation model found", fg="red"))
        click.echo("\nPlease run the pipeline first:")
        click.echo("  python -m src.orchestration.cli run --input data/raw/movies.csv")
        sys.exit(1)


def display_movie_info(movie: pd.Series, prefix: str = "Selected Movie"):
    """Print formatted movie details."""
    click.echo(f"\n{click.style(f'🎬 {prefix}:', fg='cyan', bold=True)}")
    click.echo(f"  {click.style(movie['title'], fg='white', bold=True)}")
    click.echo(f"  {click.style(movie['genres'], fg='yellow')}\n")


def display_recommendations(
    recommendations: List[Dict[str, Any]], df: pd.DataFrame, top_n: int
):
    """Print formatted recommendations."""
    click.echo(click.style(f"✨ Top {top_n} Recommendations:", fg="cyan", bold=True))
    click.echo()

    for rec in recommendations:
        similar_movie_id = rec["item_id"]
        if similar_movie_id not in df["movieId"].values:
            continue

        similar_movie = df[df["movieId"] == similar_movie_id].iloc[0]
        score = rec["score"]
        rank = rec["rank"]

        click.echo(
            f"  {click.style(f'{rank}.', fg='green', bold=True)} {click.style(similar_movie['title'], fg='white', bold=True)} {click.style(f'(ID: {similar_movie_id})', fg='blue')}"
        )
        click.echo(
            f"     {click.style('Score:', fg='cyan')} {score:.4f} | {click.style(similar_movie['genres'], fg='yellow')}"
        )
    click.echo()


def display_search_results(results: pd.DataFrame):
    """Print formatted search results."""
    if len(results) == 0:
        click.echo(click.style("No movies found", fg="yellow"))
        return

    click.echo(
        f"\n{click.style('Found ' + str(len(results)) + ' movie(s):', fg='cyan', bold=True)}\n"
    )
    click.echo(click.style("-" * 60, fg="cyan"))

    for _, row in results.iterrows():
        click.echo(
            f"  {click.style(str(row['movieId']).rjust(6), fg='green', bold=True)}:"
            f" {click.style(row['title'], fg='white')}"
        )
        click.echo(f"          {click.style(row['genres'], fg='yellow')}")

    click.echo(click.style("-" * 60, fg="cyan"))
    click.echo(
        f"\n{click.style('💡 Tip:', fg='cyan')} Use movie ID with 'recommend' command or enter ID directly"
    )


# --- Commands ---


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--version", type=str, help="Model version to use")
def cli(ctx, version):
    """🎬 Movie Recommendation System - Interactive REPL"""
    if ctx.invoked_subcommand is None:
        _run_repl(version)


@cli.command()
@click.pass_context
def version(ctx):
    """Show the current model version."""
    version = ctx.parent.params.get("version")
    registry_config = get_registry_config()
    registry = ArtifactsRegistry(registry_config)

    try:
        if version:
            registry.load_version(version)
        else:
            registry.load_latest()
        click.echo(f"Model Version: {click.style(registry.active_version, fg='cyan')}")
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)


def _run_repl(version: Optional[str]):
    """Interactive REPL mode - search and get recommendations."""
    df = load_movies()

    click.echo(click.style("Loading recommendation model...", fg="cyan"))
    recommender, _ = load_recommender_system(version)

    _print_welcome_message()

    while True:
        try:
            user_input = click.prompt(click.style("🔍", fg="cyan"), type=str).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        if user_input.lower() == "help":
            _print_help_message()
            continue

        _handle_user_input(user_input, df, recommender)

    click.echo(click.style("\nGoodbye! 👋", fg="green"))


def _print_welcome_message():
    click.echo(click.style("\n" + "-" * 60, fg="cyan"))
    click.echo(click.style("🎬  MOVIE RECOMMENDATION SYSTEM", fg="cyan", bold=True))
    click.echo(click.style("-" * 60, fg="cyan"))
    click.echo(f"\n{click.style('Commands:', fg='yellow', bold=True)}")
    click.echo("  • Type a movie name to search")
    click.echo("  • Type a movie ID number to get recommendations")
    click.echo("  • Type 'help' for more options")
    click.echo("  • Type 'quit' or press Ctrl+C to exit\n")


def _print_help_message():
    click.echo(f"\n{click.style('Available Commands:', fg='cyan', bold=True)}")
    click.echo("  • Search: Type any text to search movie titles")
    click.echo("  • Recommend: Type a movie ID (number) to get recommendations")
    click.echo("  • Help: Show this help message")
    click.echo("  • Quit: Exit the system\n")


def _handle_user_input(
    user_input: str, df: pd.DataFrame, recommender: ContentBasedRecommender
):
    # Try to parse as movie ID
    try:
        movie_id = int(user_input)
        _handle_recommendation_request(movie_id, df, recommender)
    except ValueError:
        # Treat as search term
        _handle_search_request(user_input, df)


def _handle_recommendation_request(
    movie_id: int, df: pd.DataFrame, recommender: ContentBasedRecommender
):
    if movie_id not in df["movieId"].values:
        click.echo(click.style(f"\n✗ Movie ID {movie_id} not found\n", fg="red"))
        return

    movie = df[df["movieId"] == movie_id].iloc[0]
    display_movie_info(movie, prefix="Selected")

    recommendations = recommender.recommend(movie_id, k=5)

    if not recommendations:
        click.echo(click.style(f"\n✗ Movie not in model\n", fg="red"))
        return

    display_recommendations(recommendations, df, top_n=5)


def _handle_search_request(search_term: str, df: pd.DataFrame):
    filtered = df[df["title"].str.contains(search_term, case=False, na=False)].head(15)

    if len(filtered) == 0:
        click.echo(
            click.style(f"\n✗ No movies found matching '{search_term}'\n", fg="yellow")
        )
    else:
        display_search_results(filtered)


if __name__ == "__main__":
    cli()
