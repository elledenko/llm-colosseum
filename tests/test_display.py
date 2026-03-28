"""Tests for the DisplayManager TUI module."""
import os
from unittest.mock import patch

import pytest

from eval.display import (
    DisplayManager,
    GameState,
    PlayerState,
    _bar_text,
    _is_interactive,
    get_display,
    set_display,
)


class TestPlayerState:
    """Tests for PlayerState dataclass."""

    def test_default_values(self):
        ps = PlayerState()
        assert ps.nickname == ""
        assert ps.model == ""
        assert ps.health == 176
        assert ps.super_bar == 0
        assert ps.move_count == 0
        assert ps.is_thinking is False
        assert ps.llm_buffer == ""
        assert len(ps.move_log) == 0

    def test_move_log_bounded(self):
        ps = PlayerState()
        for i in range(20):
            ps.move_log.append(f"move_{i}")
        assert len(ps.move_log) == 12  # _MAX_MOVE_LOG


class TestDisplayManagerDisabled:
    """Tests for DisplayManager in disabled mode (non-interactive)."""

    def test_disabled_manager_does_not_crash(self):
        dm = DisplayManager(enabled=False)
        dm.start()
        dm.set_player_info(1, "Test", "openai:gpt-4o", "text")
        dm.set_player_info(2, "Test2", "anthropic:claude-3", "vision")
        dm.update_health(100, 80)
        dm.update_super_bar(30, 60)
        dm.update_reward(1.5)
        dm.log_move(1, "fireball")
        dm.log_move(2, "medium punch")
        dm.set_thinking(1, True)
        dm.set_thinking(1, False)
        dm.append_llm_token(1, "- Fire")
        dm.clear_llm_buffer(1)
        dm.log_status("Game starting")
        dm.set_winner("P1")
        dm.refresh()
        dm.stop()

    def test_state_updates_correctly(self):
        dm = DisplayManager(enabled=False)
        dm.update_health(50, 75)
        assert dm._state.player1.health == 50
        assert dm._state.player2.health == 75

    def test_super_bar_updates(self):
        dm = DisplayManager(enabled=False)
        dm.update_super_bar(120, 30)
        assert dm._state.player1.super_bar == 120
        assert dm._state.player2.super_bar == 30

    def test_reward_updates(self):
        dm = DisplayManager(enabled=False)
        dm.update_reward(-2.5)
        assert dm._state.reward == -2.5

    def test_move_logging(self):
        dm = DisplayManager(enabled=False)
        dm.log_move(1, "fireball")
        dm.log_move(1, "medium punch")
        dm.log_move(2, "hurricane")
        assert dm._state.player1.move_count == 2
        assert dm._state.player2.move_count == 1
        assert list(dm._state.player1.move_log) == ["fireball", "medium punch"]
        assert list(dm._state.player2.move_log) == ["hurricane"]

    def test_thinking_state(self):
        dm = DisplayManager(enabled=False)
        dm.set_thinking(1, True)
        assert dm._state.player1.is_thinking is True
        dm.set_thinking(1, False)
        assert dm._state.player1.is_thinking is False

    def test_llm_buffer(self):
        dm = DisplayManager(enabled=False)
        dm.append_llm_token(2, "- Fire")
        dm.append_llm_token(2, "ball")
        assert dm._state.player2.llm_buffer == "- Fireball"
        dm.clear_llm_buffer(2)
        assert dm._state.player2.llm_buffer == ""

    def test_winner_state(self):
        dm = DisplayManager(enabled=False)
        dm.set_winner("P2")
        assert dm._state.winner == "P2"

    def test_player_info(self):
        dm = DisplayManager(enabled=False)
        dm.set_player_info(1, "Alice", "openai:gpt-4o", "text")
        assert dm._state.player1.nickname == "Alice"
        assert dm._state.player1.model == "openai:gpt-4o"
        assert dm._state.player1.robot_type == "text"


class TestDisplayManagerRendering:
    """Tests for the rendering methods (don't need Live to work)."""

    def test_render_returns_panel(self):
        dm = DisplayManager(enabled=False)
        dm.set_player_info(1, "P1", "openai:gpt-4o", "text")
        dm.set_player_info(2, "P2", "anthropic:claude-3", "vision")
        dm.update_health(150, 100)
        dm.update_super_bar(30, 0)
        dm.update_reward(1.0)
        dm.log_move(1, "fireball")
        panel = dm._render()
        # Should return a Panel (Rich renderable)
        from rich.panel import Panel
        assert isinstance(panel, Panel)

    def test_render_with_winner(self):
        dm = DisplayManager(enabled=False)
        dm.set_player_info(1, "Alice", "openai:gpt-4o", "text")
        dm.set_player_info(2, "Bob", "anthropic:claude-3", "text")
        dm.set_winner("P1")
        panel = dm._render()
        from rich.panel import Panel
        assert isinstance(panel, Panel)

    def test_build_summary(self):
        dm = DisplayManager(enabled=False)
        dm.set_player_info(1, "Alice", "openai:gpt-4o", "text")
        dm.set_player_info(2, "Bob", "anthropic:claude-3", "vision")
        dm.update_health(50, 0)
        dm.log_move(1, "fireball")
        dm.log_move(1, "punch")
        dm.log_move(2, "kick")
        dm.set_winner("P1")
        panel = dm._build_summary(dm._state)
        from rich.panel import Panel
        assert isinstance(panel, Panel)


class TestBarText:
    """Tests for the _bar_text helper."""

    def test_full_bar(self):
        bar = _bar_text(1.0, 10, "red")
        assert "█" * 10 in bar.plain

    def test_empty_bar(self):
        bar = _bar_text(0.0, 10, "red")
        assert "░" * 10 in bar.plain

    def test_half_bar(self):
        bar = _bar_text(0.5, 10, "red")
        plain = bar.plain
        assert plain.count("█") == 5
        assert plain.count("░") == 5


class TestIsInteractive:
    """Tests for _is_interactive detection."""

    def test_ci_env_disables(self):
        with patch.dict(os.environ, {"CI": "true"}):
            assert _is_interactive() is False

    def test_no_tui_env_disables(self):
        with patch.dict(os.environ, {"LLM_COLOSSEUM_NO_TUI": "1"}):
            assert _is_interactive() is False


class TestGlobalDisplay:
    """Tests for the module-level get/set display singleton."""

    def test_set_and_get(self):
        dm = DisplayManager(enabled=False)
        set_display(dm)
        assert get_display() is dm
        # Clean up
        set_display(None)
        assert get_display() is None


class TestThreadSafety:
    """Basic thread-safety smoke tests."""

    def test_concurrent_updates_dont_crash(self):
        import threading

        dm = DisplayManager(enabled=False)
        errors = []

        def update_p1():
            try:
                for i in range(50):
                    dm.log_move(1, f"move_{i}")
                    dm.update_health(176 - i, 176)
                    dm.set_thinking(1, i % 2 == 0)
                    dm.append_llm_token(1, f"t{i}")
            except Exception as e:
                errors.append(e)

        def update_p2():
            try:
                for i in range(50):
                    dm.log_move(2, f"move_{i}")
                    dm.update_health(176, 176 - i)
                    dm.set_thinking(2, i % 2 == 0)
                    dm.append_llm_token(2, f"t{i}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=update_p1)
        t2 = threading.Thread(target=update_p2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []
        assert dm._state.player1.move_count == 50
        assert dm._state.player2.move_count == 50
