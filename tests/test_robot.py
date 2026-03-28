"""Tests for the Robot classes: act(), observe(), context_prompt(), plan()."""
import os
from unittest.mock import patch

import numpy as np
import pytest

from agent.observer import KEN_RED, KEN_GREEN
from agent.config import MOVES, META_INSTRUCTIONS_WITH_LOWER
from agent.robot import TextRobot, VisionRobot


def _make_robot(cls=TextRobot, side=0, **kwargs):
    """Create a robot instance with default test configuration."""
    defaults = dict(
        action_space=None,
        character="Ken",
        side=side,
        character_color=KEN_RED if side == 0 else KEN_GREEN,
        ennemy_color=KEN_GREEN if side == 0 else KEN_RED,
        model="openai:gpt-4o",
        player_nb=1 if side == 0 else 2,
    )
    defaults.update(kwargs)
    return cls(**defaults)


def _make_observation(char_x=100, enemy_x=200, health=100, super_bar=0, side=0):
    """Create a mock game observation dict."""
    frame = np.zeros((224, 384, 3), dtype=np.uint8)
    # Place character and enemy colors
    if char_x is not None:
        color = KEN_RED if side == 0 else KEN_GREEN
        frame[150, char_x] = color
    if enemy_x is not None:
        color = KEN_GREEN if side == 0 else KEN_RED
        frame[150, enemy_x] = color

    p1_data = {"health": [health], "super_bar": [super_bar], "wins": [0]}
    p2_data = {"health": [health], "super_bar": [super_bar], "wins": [0]}
    return {
        "frame": frame,
        "P1": p1_data,
        "P2": p2_data,
    }


class TestRobotInit:
    """Tests for Robot initialization."""

    def test_left_side_faces_right(self):
        robot = _make_robot(side=0)
        assert robot.current_direction == "Right"

    def test_right_side_faces_left(self):
        robot = _make_robot(side=1)
        assert robot.current_direction == "Left"

    def test_default_empty_state(self):
        robot = _make_robot()
        assert robot.observations == []
        assert robot.next_steps == []
        assert robot.previous_actions == {}
        assert robot.character == "Ken"


class TestRobotAct:
    """Tests for the act() method."""

    def test_returns_zero_when_no_steps(self):
        robot = _make_robot()
        assert robot.act() == 0

    def test_returns_zero_when_sleepy(self):
        robot = _make_robot(sleepy=True)
        robot.next_steps = [5, 3, 1]
        assert robot.act() == 0

    def test_pops_first_action(self):
        robot = _make_robot()
        robot.next_steps = [5, 3, 1]
        assert robot.act() == 5
        assert robot.next_steps == [3, 1]

    def test_drains_actions_one_by_one(self):
        robot = _make_robot()
        robot.next_steps = [1, 2, 3]
        assert robot.act() == 1
        assert robot.act() == 2
        assert robot.act() == 3
        assert robot.act() == 0  # empty now

    def test_only_punch_extends_hadouken_right(self):
        robot = _make_robot(only_punch=True, side=0)
        robot.next_steps = [0]  # needs at least one step to not return 0
        first = robot.act()
        # Should have extended with Hadouken inputs
        assert len(robot.next_steps) >= 3  # Down, Right+Down, Right, High Punch minus the pop

    def test_only_punch_extends_hadouken_left(self):
        robot = _make_robot(only_punch=True, side=1)
        robot.next_steps = [0]
        first = robot.act()
        assert len(robot.next_steps) >= 3


class TestTextRobotObserve:
    """Tests for TextRobot.observe()."""

    def test_stores_observations(self):
        robot = _make_robot()
        obs = _make_observation()
        robot.observe(obs, {}, 0.0)
        assert len(robot.observations) == 1

    def test_limits_observations_to_10(self):
        robot = _make_robot()
        for i in range(15):
            obs = _make_observation()
            robot.observe(obs, {}, 0.0)
        assert len(robot.observations) == 10

    def test_detects_character_positions(self):
        robot = _make_robot(side=0)
        obs = _make_observation(char_x=100, enemy_x=300, side=0)
        robot.observe(obs, {}, 0.0)
        assert obs["character_position"] is not None
        assert obs["ennemy_position"] is not None

    def test_updates_direction_based_on_positions(self):
        robot = _make_robot(side=0)
        # Character at 100, enemy at 300 -> should face Right
        obs = _make_observation(char_x=100, enemy_x=300, side=0)
        robot.observe(obs, {}, 0.0)
        assert robot.current_direction == "Right"

    def test_tracks_previous_actions(self):
        robot = _make_robot(side=0)
        obs = _make_observation()
        robot.observe(obs, {"agent_0": 5, "agent_1": 3}, 0.0)
        assert robot.previous_actions["agent_0"] == [5]
        assert robot.previous_actions["agent_1"] == [3]

    def test_ignores_zero_actions(self):
        robot = _make_robot(side=0)
        obs = _make_observation()
        robot.observe(obs, {"agent_0": 0, "agent_1": 0}, 0.0)
        assert robot.previous_actions["agent_0"] == []
        assert robot.previous_actions["agent_1"] == []

    def test_limits_action_history_to_10(self):
        robot = _make_robot(side=0)
        for i in range(15):
            obs = _make_observation()
            robot.observe(obs, {"agent_0": i + 1}, 0.0)
        assert len(robot.previous_actions["agent_0"]) == 10


class TestTextRobotContextPrompt:
    """Tests for TextRobot.context_prompt()."""

    def _make_robot_with_obs(self, super_bar=0, reward=0.0, side=0):
        robot = _make_robot(side=side)
        obs = _make_observation(char_x=100, enemy_x=300, super_bar=super_bar, side=side)
        robot.observe(obs, {}, reward)
        return robot

    def test_far_away_prompt(self):
        robot = self._make_robot_with_obs()
        prompt = robot.context_prompt()
        assert "far" in prompt.lower() or "closer" in prompt.lower()

    def test_no_power_prompt_when_bar_zero(self):
        robot = self._make_robot_with_obs(super_bar=0)
        prompt = robot.context_prompt()
        assert "powerful" not in prompt.lower()

    def test_power_prompt_at_30(self):
        robot = self._make_robot_with_obs(super_bar=30)
        prompt = robot.context_prompt()
        assert "Megafireball" in prompt

    def test_very_powerful_at_120(self):
        robot = self._make_robot_with_obs(super_bar=120)
        prompt = robot.context_prompt()
        assert "Super attack 3" in prompt

    def test_winning_prompt(self):
        robot = self._make_robot_with_obs(reward=1.0)
        prompt = robot.context_prompt()
        assert "winning" in prompt.lower()

    def test_losing_prompt(self):
        robot = self._make_robot_with_obs(reward=-1.0)
        prompt = robot.context_prompt()
        assert "losing" in prompt.lower()


class TestVisionRobotObserve:
    """Tests for VisionRobot.observe()."""

    def test_stores_observations(self):
        robot = _make_robot(cls=VisionRobot, side=0)
        obs = _make_observation()
        robot.observe(obs, {}, 0.0)
        assert len(robot.observations) == 1

    def test_limits_observations_to_10(self):
        robot = _make_robot(cls=VisionRobot, side=0)
        for i in range(15):
            obs = _make_observation()
            robot.observe(obs, {}, 0.0)
        assert len(robot.observations) == 10

    def test_updates_direction(self):
        robot = _make_robot(cls=VisionRobot, side=0)
        obs = _make_observation(char_x=300, enemy_x=100, side=0)
        robot.observe(obs, {}, 0.0)
        assert robot.current_direction == "Left"


class TestVisionRobotImageNode:
    """Tests for VisionRobot.last_image_to_image_node()."""

    def test_returns_empty_node_when_no_observations(self):
        robot = _make_robot(cls=VisionRobot, side=0)
        node = robot.last_image_to_image_node()
        assert node.image is None  # empty ImageNode has no image data

    def test_returns_base64_encoded_image(self):
        robot = _make_robot(cls=VisionRobot, side=0)
        obs = _make_observation()
        robot.observe(obs, {}, 0.0)
        node = robot.last_image_to_image_node()
        assert node.image is not None
        assert len(node.image) > 0
        assert node.image_mimetype == "image/png"


class TestGetMovesFromLlm:
    """Tests for the get_moves_from_llm method."""

    def test_disable_llm_returns_random_move(self):
        robot = _make_robot()
        with patch.dict(os.environ, {"DISABLE_LLM": "True"}):
            moves = robot.get_moves_from_llm()
        assert len(moves) == 1
        assert moves[0] in META_INSTRUCTIONS_WITH_LOWER
