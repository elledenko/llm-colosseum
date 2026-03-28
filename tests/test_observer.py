"""Tests for the observer module: color-based position detection."""
import numpy as np
import pytest

from agent.observer import KEN_RED, KEN_GREEN, detect_position_from_color


def _make_observation(height: int = 224, width: int = 384) -> dict:
    """Create a blank (black) observation frame."""
    return {"frame": np.zeros((height, width, 3), dtype=np.uint8)}


class TestDetectPositionFromColor:
    """Tests for detect_position_from_color."""

    def test_returns_none_when_color_not_found(self):
        obs = _make_observation()
        result = detect_position_from_color(obs, KEN_RED)
        assert result is None

    def test_detects_red_color_in_play_area(self):
        obs = _make_observation()
        # Place a red pixel at row 150 (within 100:200 crop), col 200
        obs["frame"][150, 200] = KEN_RED
        result = detect_position_from_color(obs, KEN_RED)
        assert result is not None
        x, y = result
        assert x == 200  # column index
        assert y == 150  # row index + 100 offset (150 - 100 + 100 = 150)

    def test_detects_green_color(self):
        obs = _make_observation()
        obs["frame"][120, 100] = KEN_GREEN
        result = detect_position_from_color(obs, KEN_GREEN)
        assert result is not None
        x, y = result
        assert x == 100
        assert y == 120  # 20 (row in cropped) + 100 = 120

    def test_ignores_color_outside_play_area(self):
        obs = _make_observation()
        # Place color at row 50, which is outside the 100:200 crop
        obs["frame"][50, 200] = KEN_RED
        result = detect_position_from_color(obs, KEN_RED)
        assert result is None

    def test_ignores_color_below_play_area(self):
        obs = _make_observation()
        # Place color at row 210, which is outside the 100:200 crop
        obs["frame"][210, 200] = KEN_RED
        result = detect_position_from_color(obs, KEN_RED)
        assert result is None

    def test_returns_first_match_when_multiple_pixels(self):
        obs = _make_observation()
        # Place red at multiple positions
        obs["frame"][110, 50] = KEN_RED
        obs["frame"][110, 300] = KEN_RED
        result = detect_position_from_color(obs, KEN_RED)
        assert result is not None
        x, _ = result
        # np.nonzero returns sorted by row then column, so first match at col 50
        assert x == 50

    def test_epsilon_controls_color_tolerance(self):
        obs = _make_observation()
        # Place a nearly-red pixel (off by 2 in each channel)
        obs["frame"][150, 200] = [246, 2, 2]

        # With default epsilon=1, should NOT match (distance ~= 2.83)
        result_strict = detect_position_from_color(obs, KEN_RED, epsilon=1)
        assert result_strict is None

        # With epsilon=5, should match
        result_loose = detect_position_from_color(obs, KEN_RED, epsilon=5)
        assert result_loose is not None
