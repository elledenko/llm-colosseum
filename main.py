"""Unified entry point for LLM Colosseum with interactive setup.

Replaces the need to edit script.py/demo.py/local.py for different configs.
Provides an interactive Rich-based setup menu before launching the game.
"""
import sys

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, Prompt
from rich.table import Table
from rich.text import Text

from agent.config import MODELS
from eval.display import DisplayManager, set_display
from eval.game import Game, Player1, Player2

logger.remove()
logger.add(sys.stdout, level="INFO")

load_dotenv()

console = Console()


def _get_all_models() -> list:
    """Flatten the MODELS dict into a sorted list of model strings."""
    all_models = []
    for provider_models in MODELS.values():
        all_models.extend(provider_models)
    return sorted(all_models)


def _show_model_menu(all_models: list) -> None:
    """Display available models in a numbered table."""
    table = Table(title="Available Models", show_lines=False)
    table.add_column("#", style="bold", width=4)
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="dim")

    for i, model in enumerate(all_models, 1):
        provider = model.split(":")[0]
        table.add_row(str(i), model, provider)

    console.print(table)


def _pick_model(prompt_text: str, all_models: list, default: str) -> str:
    """Let the user pick a model by number or type a custom model string."""
    choice = Prompt.ask(
        prompt_text,
        default=default,
        console=console,
    )
    # If they typed a number, look it up
    try:
        idx = int(choice)
        if 1 <= idx <= len(all_models):
            return all_models[idx - 1]
    except ValueError:
        pass
    # Otherwise treat it as a raw model string
    return choice


def interactive_setup() -> dict:
    """Run the interactive pre-game setup and return config dict."""
    console.print()
    console.print(
        Panel(
            Text("STREET FIGHTER III — LLM COLOSSEUM", justify="center", style="bold white"),
            border_style="bright_blue",
            subtitle="Interactive Setup",
        )
    )
    console.print()

    all_models = _get_all_models()
    _show_model_menu(all_models)
    console.print()
    console.print("[dim]Enter a number from the list above, or type a custom model string (e.g. ollama:mistral)[/dim]")
    console.print()

    # Player 1
    console.print("[bold red]── Player 1 Configuration ──[/bold red]")
    p1_model = _pick_model("  Model", all_models, default="openai:gpt-4o-mini")
    p1_type = Prompt.ask("  Mode", choices=["text", "vision"], default="text", console=console)
    p1_temp = FloatPrompt.ask("  Temperature", default=0.7, console=console)
    p1_nick = Prompt.ask("  Nickname", default="Player1", console=console)

    console.print()

    # Player 2
    console.print("[bold green]── Player 2 Configuration ──[/bold green]")
    p2_model = _pick_model("  Model", all_models, default="anthropic:claude-3-haiku-20240307")
    p2_type = Prompt.ask("  Mode", choices=["text", "vision"], default="text", console=console)
    p2_temp = FloatPrompt.ask("  Temperature", default=0.7, console=console)
    p2_nick = Prompt.ask("  Nickname", default="Player2", console=console)

    console.print()

    # Game options
    console.print("[bold]── Game Options ──[/bold]")
    render = Confirm.ask("  Render game window?", default=True, console=console)
    save_game = Confirm.ask("  Save replay?", default=False, console=console)

    console.print()

    return {
        "p1_model": p1_model,
        "p1_type": p1_type,
        "p1_temp": p1_temp,
        "p1_nick": p1_nick,
        "p2_model": p2_model,
        "p2_type": p2_type,
        "p2_temp": p2_temp,
        "p2_nick": p2_nick,
        "render": render,
        "save_game": save_game,
    }


def run_game(config: dict):
    """Launch the game with the given configuration."""
    # Initialize display manager before creating players
    display = DisplayManager()
    set_display(display)

    game = Game(
        render=config["render"],
        save_game=config["save_game"],
        player_1=Player1(
            nickname=config["p1_nick"],
            model=config["p1_model"],
            robot_type=config["p1_type"],
            temperature=config["p1_temp"],
        ),
        player_2=Player2(
            nickname=config["p2_nick"],
            model=config["p2_model"],
            robot_type=config["p2_type"],
            temperature=config["p2_temp"],
        ),
    )
    return game.run()


def main():
    """Interactive main entry point."""
    config = interactive_setup()

    # Show summary before starting
    console.print(
        Panel(
            f"[red]{config['p1_nick']}[/red] ({config['p1_model']}) [bold]vs[/bold] "
            f"[green]{config['p2_nick']}[/green] ({config['p2_model']})",
            title="[bold]Match Setup[/bold]",
            border_style="bright_yellow",
        )
    )

    if not Confirm.ask("Start the match?", default=True, console=console):
        console.print("[dim]Match cancelled.[/dim]")
        return

    console.print()
    return run_game(config)


if __name__ == "__main__":
    main()
