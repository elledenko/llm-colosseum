"""Tests for the config module: moves, combos, and meta-instructions."""
import pytest

from agent.config import (
    COMBOS,
    INDEX_TO_MOVE,
    META_INSTRUCTIONS,
    META_INSTRUCTIONS_WITH_LOWER,
    MODELS,
    MOVES,
    MOVES_WITH_LOWER,
    NB_FRAME_WAIT,
    REAL_MOVE_LIST,
    SPECIAL_MOVES,
    X_SIZE,
    Y_SIZE,
)


class TestMoves:
    """Tests for the MOVES dictionary."""

    def test_no_move_is_zero(self):
        assert MOVES["No-Move"] == 0

    def test_all_move_values_are_non_negative(self):
        for name, value in MOVES.items():
            assert value >= 0, f"Move '{name}' has negative value {value}"

    def test_directional_aliases_are_consistent(self):
        """Left+Up and Up+Left should map to the same action."""
        assert MOVES["Left+Up"] == MOVES["Up+Left"]
        assert MOVES["Up+Right"] == MOVES["Right+Up"]
        assert MOVES["Right+Down"] == MOVES["Down+Right"]
        assert MOVES["Down+Left"] == MOVES["Left+Down"]

    def test_moves_with_lower_contains_lowercase_versions(self):
        for key in MOVES:
            assert key.lower() in MOVES_WITH_LOWER
            assert MOVES_WITH_LOWER[key.lower()] == MOVES[key]


class TestIndexToMove:
    """Tests for INDEX_TO_MOVE reverse mapping."""

    def test_covers_all_unique_values(self):
        unique_values = set(MOVES.values())
        for v in unique_values:
            assert v in INDEX_TO_MOVE, f"Value {v} missing from INDEX_TO_MOVE"

    def test_maps_back_to_valid_move_names(self):
        for idx, name in INDEX_TO_MOVE.items():
            assert name in MOVES


class TestMetaInstructions:
    """Tests for META_INSTRUCTIONS."""

    def test_all_meta_instructions_have_left_and_right(self):
        for name, directions in META_INSTRUCTIONS.items():
            assert "left" in directions, f"'{name}' missing 'left' key"
            assert "right" in directions, f"'{name}' missing 'right' key"

    def test_meta_instructions_with_lower_has_all_keys(self):
        for key in META_INSTRUCTIONS:
            assert key.lower() in META_INSTRUCTIONS_WITH_LOWER

    def test_move_closer_goes_forward(self):
        """Move Closer should move toward the opponent."""
        # When on the left side facing right, "Move Closer" should go right (action 5)
        assert all(a == 5 for a in META_INSTRUCTIONS["Move Closer"]["right"])
        # When on the right side facing left, should go left (action 1)
        assert all(a == 1 for a in META_INSTRUCTIONS["Move Closer"]["left"])

    def test_move_away_goes_backward(self):
        """Move Away should move away from the opponent."""
        assert all(a == 1 for a in META_INSTRUCTIONS["Move Away"]["right"])
        assert all(a == 5 for a in META_INSTRUCTIONS["Move Away"]["left"])


class TestCombos:
    """Tests for COMBOS dictionary."""

    def test_fireball_combo_has_correct_length(self):
        assert len(COMBOS["Fireball (Hadouken)"]["right"]) == 4
        assert len(COMBOS["Fireball (Hadouken)"]["left"]) == 4

    def test_all_combo_actions_are_valid_move_ids(self):
        all_move_ids = set(MOVES.values())
        for combo_name, directions in COMBOS.items():
            for direction, actions in directions.items():
                for action in actions:
                    assert action in all_move_ids, (
                        f"Combo '{combo_name}' ({direction}) has invalid action {action}"
                    )


class TestSpecialMoves:
    """Tests for SPECIAL_MOVES dictionary."""

    def test_ex_moves_are_longer_than_base_combos(self):
        assert len(SPECIAL_MOVES["EX-Fireball (Hadouken)"]["right"]) > len(
            COMBOS["Fireball (Hadouken)"]["right"]
        )

    def test_all_special_move_actions_are_valid(self):
        all_move_ids = set(MOVES.values())
        for name, directions in SPECIAL_MOVES.items():
            for direction, actions in directions.items():
                for action in actions:
                    assert action in all_move_ids, (
                        f"Special move '{name}' ({direction}) has invalid action {action}"
                    )


class TestRealMoveList:
    """Tests for REAL_MOVE_LIST."""

    def test_real_move_list_starts_with_no_move(self):
        assert REAL_MOVE_LIST[0] == "No-Move"

    def test_real_move_list_has_all_directions(self):
        assert "Up+Right" in REAL_MOVE_LIST
        assert "Left" in REAL_MOVE_LIST
        assert "Right" in REAL_MOVE_LIST
        assert "Down" in REAL_MOVE_LIST


class TestConstants:
    """Tests for game constants."""

    def test_game_dimensions(self):
        assert X_SIZE == 384
        assert Y_SIZE == 224

    def test_frame_wait_positive(self):
        assert NB_FRAME_WAIT >= 0
