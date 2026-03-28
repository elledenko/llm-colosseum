import datetime
import os
import random
import traceback
from threading import Thread
from typing import List, Optional

from agent import KEN_GREEN, KEN_RED, TextRobot, VisionRobot
from agent.config import MODELS
from agent.robot import Robot
from diambra.arena import (
    EnvironmentSettingsMultiAgent,
    RecordingSettings,
    SpaceTypes,
    make,
)
from loguru import logger

from eval.display import DisplayManager, get_display, set_display


def generate_random_model(openai: bool = False, mistral: bool = True):
    models_available = []

    for model, models in MODELS.items():
        if openai and model == "OPENAI":
            models_available.extend(models)
        if mistral and model == "MISTRAL":
            models_available.extend(models)

    random.seed()
    # Generate a pair of random two models
    random_model = random.choice(models_available)

    return random_model


class Player:
    nickname: str
    model: str
    robot: Optional[Robot] = None
    temperature: float = 0.7

    # Map provider prefixes to their required environment variable
    _PROVIDER_API_KEYS = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
        "anyscale": "ANYSCALE_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
    }

    def verify_provider_name(self):
        """Verify that the required API key is set for the model's provider."""
        provider = self.model.split(":")[0]
        env_var = self._PROVIDER_API_KEYS.get(provider)
        if env_var is not None:
            assert os.environ.get(env_var) is not None, (
                f"{provider.capitalize()} API key not set. "
                f"Please set the {env_var} environment variable."
            )


VALID_ROBOT_TYPES = {"text", "vision"}


class Player1(Player):
    def __init__(
        self,
        nickname: str,
        model: str,
        robot_type: str = "text",
        temperature: float = 0.7,
    ):
        if robot_type not in VALID_ROBOT_TYPES:
            raise ValueError(
                f"Invalid robot_type '{robot_type}'. Must be one of: {sorted(VALID_ROBOT_TYPES)}"
            )
        self.nickname = nickname
        self.model = model
        self.robot_type = robot_type
        self.temperature = temperature

        if robot_type == "vision":
            self.robot = VisionRobot(
                action_space=None,
                character="Ken",
                side=0,
                character_color=KEN_RED,
                model=self.model,
                ennemy_color=KEN_GREEN,
                only_punch=os.getenv("TEST_MODE", False),
                temperature=self.temperature,
                sleepy=False,
                player_nb=1,
            )
        else:
            self.robot = TextRobot(
                action_space=None,
                character="Ken",
                side=0,
                character_color=KEN_RED,
                ennemy_color=KEN_GREEN,
                only_punch=os.getenv("TEST_MODE", False),
                temperature=self.temperature,
                sleepy=False,
                model=self.model,
                player_nb=1,
            )
        display = get_display()
        if display:
            display.set_player_info(1, self.nickname, self.model, self.robot_type)
        else:
            logger.info(f"Player 1 using: {self.model}")
        self.verify_provider_name()


class Player2(Player):
    def __init__(
        self,
        nickname: str,
        model: str,
        robot_type: str = "text",
        temperature: float = 0.7,
    ):
        if robot_type not in VALID_ROBOT_TYPES:
            raise ValueError(
                f"Invalid robot_type '{robot_type}'. Must be one of: {sorted(VALID_ROBOT_TYPES)}"
            )
        self.nickname = nickname
        self.model = model
        self.robot_type = robot_type
        self.temperature = temperature
        if robot_type == "vision":
            self.robot = VisionRobot(
                action_space=None,
                character="Ken",
                side=1,
                model=self.model,
                character_color=KEN_GREEN,
                ennemy_color=KEN_RED,
                temperature=self.temperature,
                sleepy=os.getenv("TEST_MODE", False),
                player_nb=2,
            )
        else:
            self.robot = TextRobot(
                action_space=None,
                character="Ken",
                side=1,
                character_color=KEN_GREEN,
                ennemy_color=KEN_RED,
                temperature=self.temperature,
                sleepy=os.getenv("TEST_MODE", False),
                model=self.model,
                player_nb=2,
            )
        display = get_display()
        if display:
            display.set_player_info(2, self.nickname, self.model, self.robot_type)
        else:
            logger.info(f"Player 2 using: {self.model}")
        self.verify_provider_name()


class Episode:
    player_1: Optional[Player1]
    player_2: Player2
    player_1_won: Optional[bool] = None

    def __init__(self, player_1: Optional[Player1], player_2: Player2):
        self.player_1 = player_1
        self.player_2 = player_2

    def save(self):
        if self.player_1 is None:
            # Playing with the controller
            # Results are not saved
            return

        # Write the results to an existing csv with headers "player_1", "player_2", "winner"
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Verify if the file exists
        if not os.path.exists("results.csv"):
            with open("results.csv", "w") as f:
                f.write(
                    "id,player_1_model,player_1_robot_type,player_1_temperature,player_2_model,player_2_robot_type,player_2_temperature,player_1_won\n"
                )

        with open("results.csv", "a") as f:
            f.write(
                f"{timestamp},{self.player_1.model},{self.player_1.robot_type},{self.player_1.temperature},"
                + f"{self.player_2.model},{self.player_2.robot_type},{self.player_2.temperature},{self.player_1_won}\n"
            )


class Game:
    player_1: Optional[Player1] = None  # First player. None if Human
    player_2: Player2

    render: Optional[bool] = False
    splash_screen: Optional[bool] = False
    save_game: Optional[bool] = False
    characters: Optional[List[str]] = ["Ken", "Ken"]
    outfits: Optional[List[int]] = [1, 3]
    frame_shape: Optional[List[int]] = [0, 0, 0]
    seed: Optional[int] = 42
    settings: EnvironmentSettingsMultiAgent = None  # Settings of the game
    env = None  # Environment of the game

    def __init__(
        self,
        player_1: Optional[Player1],
        player_2: Player2,
        render: bool = False,
        save_game: bool = False,
        splash_screen: bool = False,
        characters: List[str] = ["Ken", "Ken"],
        super_arts: List[int] = [3, 3],
        outfits: List[int] = [1, 3],
        frame_shape: List[int] = [0, 0, 0],
        seed: int = 42,
    ):
        """_summary_

        Args:
            render (bool, optional): Renders the fights. Defaults to False.
            splash_screen (bool, optional): Display the splash screen. Defaults to False.
            characters (List[str], optional): List of the players to have. Defaults to ["Ryu", "Ken"].
            outfits (List[int], optional): Outfits to run. Defaults to [2, 2].
            frame_shape (List[int], optional): Don't know :D . Defaults to [0, 0, 0].
            seed (int, optional): Random seed. Defaults to 42.
        """
        self.render = render
        self.splash_screen = splash_screen
        self.save_game = save_game
        self.characters = characters
        self.super_arts = super_arts
        self.outfits = outfits
        self.frame_shape = frame_shape
        self.seed = seed
        self.settings = self._init_settings()
        self.env = self._init_env(self.settings)
        self.observation, self.info = self.env.reset(seed=self.seed)
        if player_1 is not None:
            self.player_1 = player_1
        else:
            # If player 1 is not provided, we will use the controller
            # The human player will be able to play with the controller
            from diambra.arena.utils.controller import get_diambra_controller

            self.player_1 = None
            self.controller = get_diambra_controller(
                self.env.unwrapped.get_actions_tuples(),
                # force_configure=True,
            )
            self.controller.start()
        self.player_2 = player_2

    def _init_settings(self) -> EnvironmentSettingsMultiAgent:
        """
        Initializes the settings for the game.
        """
        settings = EnvironmentSettingsMultiAgent(
            render_mode="rgb_array",
            splash_screen=self.splash_screen,
        )

        settings.action_space = (SpaceTypes.DISCRETE, SpaceTypes.DISCRETE)
        settings.characters = self.characters
        settings.outfits = self.outfits
        settings.frame_shape = self.frame_shape
        settings.super_art = self.super_arts

        return settings

    def _init_recorder(self) -> RecordingSettings:
        """
        Initializes the recorder for the game.
        """
        if not self.save_game:
            return None
        # Recording settings in root directory
        root_dir = os.path.dirname(os.path.abspath(__file__))
        game_id = "sfiii3n"
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        recording_settings = RecordingSettings()
        recording_settings.dataset_path = os.path.join(
            root_dir, "diambra/episode_recording", game_id, "-", timestamp
        )
        recording_settings.username = "llm-colosseum"

        return recording_settings

    def _init_env(self, settings: EnvironmentSettingsMultiAgent):
        """
        Initializes the environment for the game.
        """
        render_mode = "human" if self.render else "rgb_array"
        recorder_settings = self._init_recorder()
        if self.save_game:
            return make(
                "sfiii3n",
                settings,
                render_mode=render_mode,
                episode_recording_settings=recorder_settings,
            )
        return make("sfiii3n", settings, render_mode=render_mode)

    def _save(self):
        """
        Save the game state.
        """
        pass

    def _determine_winner(self, episode: Episode) -> Optional[bool]:
        """Determine the winner based on remaining health.

        Returns True if player 1 won, False if player 2 won, None for a draw.
        Also sets episode.player_1_won.
        """
        p1_health = self.observation["P1"]["health"][0]
        p2_health = self.observation["P2"]["health"][0]
        if p1_health > p2_health:
            episode.player_1_won = True
        elif p2_health > p1_health:
            episode.player_1_won = False
        else:
            episode.player_1_won = None
        return episode.player_1_won

    def run(self):
        """
        Runs the game with the given settings.
        """
        display = get_display()
        if display:
            display.start()

        try:
            self.actions = {
                "agent_0": 0,
                "agent_1": 0,
            }
            self.reward = 0.0

            if self.player_1 is not None:
                self.player_1.robot.observe(self.observation, {}, 0.0)

            self.player_2.robot.observe(self.observation, {}, 0.0)
            # Initialize the episode
            episode = Episode(player_1=self.player_1, player_2=self.player_2)
            # Start the threads that make API calls
            if self.player_1 is not None:
                player1_thread = PlanAndActPlayer1(game=self, episode=episode)
                player1_thread.start()
            player2_thread = PlanAndActPlayer2(game=self, episode=episode)
            player2_thread.start()

            while True:
                # Render the game
                if self.render:
                    self.env.render()

                actions = self.actions

                if self.player_1 is None:
                    # If player 1 is not provided, we use the controller
                    try:
                        controller_actions = self.controller.get_actions()
                        actions["agent_1"] = (
                            controller_actions[0] + controller_actions[1]
                        )
                    except Exception as e:
                        logger.debug(f"Controller error: {e}")

                if "agent_0" not in actions:
                    actions["agent_0"] = 0
                if "agent_1" not in actions:
                    actions["agent_1"] = 0
                observation, reward, terminated, truncated, info = self.env.step(
                    actions
                )
                # Remove the actions that were executed
                if "agent_0" in self.actions:
                    del actions["agent_0"]
                if "agent_1" in self.actions:
                    del actions["agent_1"]

                self.observation = observation
                self.reward += reward

                # Feed live game state to the display manager
                if display:
                    display.update_health(
                        observation["P1"]["health"][0],
                        observation["P2"]["health"][0],
                    )
                    display.update_super_bar(
                        observation["P1"]["super_bar"][0],
                        observation["P2"]["super_bar"][0],
                    )
                    display.update_reward(self.reward)
                    display.refresh()

                p1_wins = observation["P1"]["wins"][0]
                p2_wins = observation["P2"]["wins"][0]

                if p1_wins == 1 or p2_wins == 1:
                    if self.player_1 is not None:
                        player1_thread.running = False
                    player2_thread.running = False
                    episode.player_1_won = p1_wins == 1

                    winner_tag = "P1" if episode.player_1_won else "P2"
                    if display:
                        display.set_winner(winner_tag)
                        display.refresh()
                    else:
                        if episode.player_1_won:
                            logger.info(
                                f"Player1 {self.player_1.robot.model} '{self.player_1.nickname}' won!"
                            )
                        else:
                            logger.info(
                                f"Player2 {self.player_2.robot.model} '{self.player_2.nickname}' won!"
                            )

                    episode.save()
                    self.env.close()

                    # Show post-game summary
                    if display:
                        display.stop()
                        display.show_post_game_summary()

                    return episode.player_1_won
        except Exception as e:
            logger.error(f"Game exception: {e}")
            traceback.print_exc()
        finally:
            if display:
                display.stop()
            try:
                if self.player_1 is None:
                    self.controller.stop()
                self.env.close()
            except Exception:
                pass  # Environment may already be closed
        return 0


class PlanAndAct(Thread):
    def __init__(self, game: Game, episode: Episode):
        self.running = True
        self.game = game
        self.episode = episode

        Thread.__init__(self, daemon=True)
        # atexit.register(self.stop)


class PlanAndActPlayer1(PlanAndAct):
    def run(self) -> None:
        while self.running:
            if "agent_0" not in self.game.actions:
                # Plan
                self.game.player_1.robot.plan()
                # Act
                self.game.actions["agent_0"] = self.game.player_1.robot.act()
                # Observe the environment
                self.game.player_1.robot.observe(
                    self.game.observation, self.game.actions, self.game.reward
                )


class PlanAndActPlayer2(PlanAndAct):
    def run(self) -> None:
        while self.running:
            if "agent_1" not in self.game.actions:
                # Plan
                self.game.player_2.robot.plan()
                # Act
                self.game.actions["agent_1"] = self.game.player_2.robot.act()
                # Observe the environment
                self.game.player_2.robot.observe(
                    self.game.observation, self.game.actions, -self.game.reward
                )
