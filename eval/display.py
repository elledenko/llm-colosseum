"""Thread-safe TUI display manager using Rich Live.

Provides a structured dashboard that replaces scattered print() calls
throughout the codebase. All game state updates flow through this
single display manager, which renders a live-updating terminal UI.

Gracefully degrades to simple logging when Rich Live is unavailable
(e.g., non-interactive terminals, CI environments, piped output).
"""
import csv
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from loguru import logger

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


# Maximum number of moves to show in the scrolling log per player
_MAX_MOVE_LOG = 12

# Maximum number of status messages to keep
_MAX_STATUS_LOG = 5


@dataclass
class PlayerState:
    """Mutable game state for one player, updated from game threads."""
    nickname: str = ""
    model: str = ""
    robot_type: str = "text"
    health: int = 176  # SF3 default max health
    max_health: int = 176
    super_bar: int = 0
    max_super_bar: int = 160
    move_log: Deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_MOVE_LOG))
    move_count: int = 0
    is_thinking: bool = False
    llm_buffer: str = ""  # buffered LLM stream text for this player


@dataclass
class GameState:
    """Complete game state snapshot for rendering."""
    player1: PlayerState = field(default_factory=PlayerState)
    player2: PlayerState = field(default_factory=PlayerState)
    reward: float = 0.0
    status_messages: Deque[str] = field(
        default_factory=lambda: deque(maxlen=_MAX_STATUS_LOG)
    )
    game_active: bool = False
    winner: Optional[str] = None  # "P1", "P2", or None


class DisplayManager:
    """Thread-safe display manager for the game TUI.

    All public methods are safe to call from any thread. Internal state
    is guarded by a lock, and the Rich Live display is only touched
    from its own refresh cycle.
    """

    def __init__(self, enabled: bool = True):
        self._lock = threading.Lock()
        self._state = GameState()
        self._console = Console()
        self._live: Optional[Live] = None
        # Decide whether to use Live TUI or fallback to simple logging
        self._enabled = enabled and _is_interactive()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "DisplayManager":
        """Start the live display. Call before the game loop begins."""
        if self._enabled:
            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=4,
                screen=False,
            )
            self._live.start()
        with self._lock:
            self._state.game_active = True
        return self

    def stop(self) -> None:
        """Stop the live display. Call after the game loop ends."""
        with self._lock:
            self._state.game_active = False
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def refresh(self) -> None:
        """Force a refresh of the live display (call from the main game loop)."""
        if self._live is not None:
            try:
                self._live.update(self._render())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Player setup (called once during init)
    # ------------------------------------------------------------------

    def set_player_info(
        self,
        player_nb: int,
        nickname: str,
        model: str,
        robot_type: str,
    ) -> None:
        with self._lock:
            ps = self._get_player(player_nb)
            ps.nickname = nickname
            ps.model = model
            ps.robot_type = robot_type
        if not self._enabled:
            color = "red" if player_nb == 1 else "green"
            logger.info(f"Player {player_nb} ({color}): {model} [{robot_type}]")

    # ------------------------------------------------------------------
    # Game state updates (called from game threads)
    # ------------------------------------------------------------------

    def update_health(self, p1_health: int, p2_health: int) -> None:
        with self._lock:
            self._state.player1.health = p1_health
            self._state.player2.health = p2_health

    def update_super_bar(self, p1_super: int, p2_super: int) -> None:
        with self._lock:
            self._state.player1.super_bar = p1_super
            self._state.player2.super_bar = p2_super

    def update_reward(self, reward: float) -> None:
        with self._lock:
            self._state.reward = reward

    def log_move(self, player_nb: int, move_name: str) -> None:
        with self._lock:
            ps = self._get_player(player_nb)
            ps.move_log.append(move_name)
            ps.move_count += 1
        if not self._enabled:
            tag = "P1" if player_nb == 1 else "P2"
            logger.info(f"[{tag}] move: {move_name}")

    def set_thinking(self, player_nb: int, thinking: bool) -> None:
        with self._lock:
            self._get_player(player_nb).is_thinking = thinking

    def append_llm_token(self, player_nb: int, token: str) -> None:
        """Buffer a streamed LLM token for a player (replaces raw print)."""
        with self._lock:
            self._get_player(player_nb).llm_buffer += token

    def clear_llm_buffer(self, player_nb: int) -> None:
        with self._lock:
            self._get_player(player_nb).llm_buffer = ""

    def log_status(self, message: str) -> None:
        with self._lock:
            self._state.status_messages.append(message)
        if not self._enabled:
            logger.info(message)

    def set_winner(self, winner: str) -> None:
        """Set the winner: 'P1' or 'P2'."""
        with self._lock:
            self._state.winner = winner

    # ------------------------------------------------------------------
    # Post-game summary
    # ------------------------------------------------------------------

    def show_post_game_summary(self) -> None:
        """Print a rich post-game summary panel after the Live display stops."""
        with self._lock:
            state = self._state

        summary = self._build_summary(state)
        self._console.print()
        self._console.print(summary)
        self._console.print()

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _get_player(self, player_nb: int) -> PlayerState:
        if player_nb == 1:
            return self._state.player1
        return self._state.player2

    def _render(self) -> Panel:
        """Build the full dashboard renderable. Called under refresh cycle."""
        with self._lock:
            # Snapshot state under lock
            p1 = self._state.player1
            p2 = self._state.player2
            reward = self._state.reward
            winner = self._state.winner
            status_msgs = list(self._state.status_messages)

        # -- Title bar --
        title = Text("STREET FIGHTER III — LLM COLOSSEUM", style="bold white on blue")

        # -- Health bars --
        health_table = self._build_health_table(p1, p2)

        # -- Super bars --
        super_table = self._build_super_table(p1, p2)

        # -- Score --
        if reward > 0:
            score_style = "bold red"
            score_hint = "P1 leading"
        elif reward < 0:
            score_style = "bold green"
            score_hint = "P2 leading"
        else:
            score_style = "bold white"
            score_hint = "Even"
        score_text = Text(
            f"Score: {reward:+.1f}  ({score_hint})", style=score_style
        )

        # -- Move logs (two-column) --
        move_panel = self._build_move_panel(p1, p2)

        # -- LLM stream buffers --
        llm_panel = self._build_llm_panel(p1, p2)

        # -- Status bar --
        if winner:
            if winner == "P1":
                status_line = Text(
                    f"  WINNER: {p1.nickname} ({p1.model})", style="bold red on white"
                )
            else:
                status_line = Text(
                    f"  WINNER: {p2.nickname} ({p2.model})", style="bold green on white"
                )
        elif status_msgs:
            status_line = Text(status_msgs[-1], style="dim")
        else:
            status_line = Text("Game in progress...", style="dim italic")

        body = Group(
            Align.center(title),
            "",
            health_table,
            super_table,
            Align.center(score_text),
            "",
            move_panel,
            "",
            llm_panel,
            "",
            status_line,
        )

        return Panel(body, title="[bold]LLM Colosseum[/bold]", border_style="bright_blue")

    def _build_health_table(self, p1: PlayerState, p2: PlayerState) -> Table:
        table = Table(show_header=False, show_edge=False, pad_edge=False, expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=3)
        table.add_column(ratio=1, justify="right")
        table.add_column(ratio=3)
        table.add_column(ratio=1, justify="right")

        p1_pct = max(0, p1.health / p1.max_health)
        p2_pct = max(0, p2.health / p2.max_health)
        p1_bar = _bar_text(p1_pct, 20, "red")
        p2_bar = _bar_text(p2_pct, 20, "green")

        p1_label = Text(f" P1 {p1.nickname}", style="bold red")
        p2_label = Text(f" P2 {p2.nickname}", style="bold green")

        table.add_row(
            p1_label,
            p1_bar,
            Text(f"{p1.health}", style="red"),
            p2_bar,
            Text(f"{p2.health}", style="green"),
        )
        # Model names below
        table.add_row(
            Text(f"  {p1.model}", style="dim red"),
            Text(""),
            Text("HP", style="dim"),
            Text(""),
            Text(f"{p2.model}  ", style="dim green"),
        )
        return table

    def _build_super_table(self, p1: PlayerState, p2: PlayerState) -> Table:
        table = Table(show_header=False, show_edge=False, pad_edge=False, expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=3)
        table.add_column(ratio=1, justify="right")
        table.add_column(ratio=3)
        table.add_column(ratio=1, justify="right")

        p1_pct = min(1.0, p1.super_bar / max(1, p1.max_super_bar))
        p2_pct = min(1.0, p2.super_bar / max(1, p2.max_super_bar))
        p1_bar = _bar_text(p1_pct, 20, "yellow")
        p2_bar = _bar_text(p2_pct, 20, "cyan")

        table.add_row(
            Text("  Super", style="dim"),
            p1_bar,
            Text(f"{p1.super_bar}", style="yellow"),
            p2_bar,
            Text(f"{p2.super_bar}", style="cyan"),
        )
        return table

    def _build_move_panel(self, p1: PlayerState, p2: PlayerState) -> Columns:
        """Build the two-column move log."""
        p1_moves = self._format_move_list(p1, "red")
        p2_moves = self._format_move_list(p2, "green")

        p1_thinking = " [dim italic](thinking...)[/]" if p1.is_thinking else ""
        p2_thinking = " [dim italic](thinking...)[/]" if p2.is_thinking else ""

        left = Panel(
            p1_moves,
            title=f"[red]P1 Moves ({p1.move_count}){p1_thinking}[/red]",
            border_style="red",
            width=40,
        )
        right = Panel(
            p2_moves,
            title=f"[green]P2 Moves ({p2.move_count}){p2_thinking}[/green]",
            border_style="green",
            width=40,
        )
        return Columns([left, right], expand=True, align="center")

    def _format_move_list(self, ps: PlayerState, color: str) -> Text:
        if not ps.move_log:
            return Text("  Waiting for first move...", style="dim italic")
        lines = Text()
        for i, move in enumerate(ps.move_log):
            prefix = ">" if i == len(ps.move_log) - 1 else " "
            style = f"bold {color}" if i == len(ps.move_log) - 1 else color
            lines.append(f" {prefix} {move}\n", style=style)
        return lines

    def _build_llm_panel(self, p1: PlayerState, p2: PlayerState) -> Columns:
        """Show buffered LLM stream tokens per player."""
        # Truncate to last 80 chars for display
        p1_buf = p1.llm_buffer[-80:] if p1.llm_buffer else "(idle)"
        p2_buf = p2.llm_buffer[-80:] if p2.llm_buffer else "(idle)"
        left = Panel(
            Text(p1_buf, style="dim red", overflow="ellipsis"),
            title="[red]P1 LLM Stream[/red]",
            border_style="dim red",
            height=4,
            width=40,
        )
        right = Panel(
            Text(p2_buf, style="dim green", overflow="ellipsis"),
            title="[green]P2 LLM Stream[/green]",
            border_style="dim green",
            height=4,
            width=40,
        )
        return Columns([left, right], expand=True, align="center")

    def _build_summary(self, state: GameState) -> Panel:
        """Build the post-game summary panel."""
        p1 = state.player1
        p2 = state.player2

        table = Table(title="Game Results", show_header=True, expand=True)
        table.add_column("", style="bold", width=20)
        table.add_column("Player 1", style="red", justify="center")
        table.add_column("Player 2", style="green", justify="center")

        table.add_row("Nickname", p1.nickname, p2.nickname)
        table.add_row("Model", p1.model, p2.model)
        table.add_row("Mode", p1.robot_type, p2.robot_type)
        table.add_row("Final HP", str(p1.health), str(p2.health))
        table.add_row("Moves", str(p1.move_count), str(p2.move_count))

        # Winner announcement
        if state.winner == "P1":
            winner_text = Text(
                f"  {p1.nickname} ({p1.model}) WINS!",
                style="bold red on white",
            )
        elif state.winner == "P2":
            winner_text = Text(
                f"  {p2.nickname} ({p2.model}) WINS!",
                style="bold green on white",
            )
        else:
            winner_text = Text("  DRAW", style="bold white on blue")

        # Historical record from results.csv
        record_text = self._load_win_record(p1.model, p2.model)

        body = Group(
            table,
            "",
            Align.center(winner_text),
            "",
            record_text,
        )
        return Panel(body, title="[bold]Post-Game Summary[/bold]", border_style="bright_yellow")

    def _load_win_record(self, p1_model: str, p2_model: str) -> Text:
        """Load historical win/loss record from results.csv if available."""
        if not os.path.exists("results.csv"):
            return Text("  No historical data (results.csv not found)", style="dim")

        try:
            p1_wins = 0
            p2_wins = 0
            total = 0
            with open("results.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    won = row.get("player_1_won", "").strip()
                    if won == "True":
                        p1_wins += 1
                    elif won == "False":
                        p2_wins += 1

            text = Text()
            text.append("  Historical Record: ", style="bold")
            text.append(f"{total} games played — ", style="dim")
            text.append(f"P1 wins: {p1_wins}", style="red")
            text.append(" | ", style="dim")
            text.append(f"P2 wins: {p2_wins}", style="green")
            text.append(f" | Draws: {total - p1_wins - p2_wins}", style="dim")
            return text
        except Exception:
            return Text("  Could not read results.csv", style="dim")


def _bar_text(pct: float, width: int, color: str) -> Text:
    """Render a simple text-based progress bar."""
    filled = int(pct * width)
    empty = width - filled
    bar = Text()
    bar.append("█" * filled, style=f"bold {color}")
    bar.append("░" * empty, style="dim")
    return bar


def _is_interactive() -> bool:
    """Check if we're in an interactive terminal that supports Rich Live."""
    if os.getenv("CI"):
        return False
    if os.getenv("LLM_COLOSSEUM_NO_TUI"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


# Module-level singleton — set by Game or main.py before threads start
_display: Optional[DisplayManager] = None
_display_lock = threading.Lock()


def get_display() -> Optional[DisplayManager]:
    """Get the global DisplayManager instance (may be None)."""
    return _display


def set_display(dm: DisplayManager) -> None:
    """Set the global DisplayManager instance."""
    global _display
    with _display_lock:
        _display = dm
