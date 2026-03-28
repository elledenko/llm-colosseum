# LLM Colosseum

<div align="center">
    <img src="./logo.png" alt="colosseum-logo" width="30%"  style="border-radius: 50%; padding-bottom: 20px"/>
</div>

Make LLMs fight each other in real time in Street Fighter III. Which model is the best fighter?

**Live demo:** [llm-colosseum.phospho.ai](https://llm-colosseum.phospho.ai)

https://github.com/OpenGenerativeAI/llm-colosseum/assets/19614572/79b58e26-7902-4687-af5d-0e1e845ecaf8

---

## Quick Start

### Prerequisites

1. **DIAMBRA** — the game engine. Follow the [installation guide](https://docs.diambra.ai/#installation).
2. **ROM** — download the Street Fighter III ROM and place it in `~/.diambra/roms` (keep it zipped).
3. **Python 3.10+** — a [virtual environment](https://docs.python.org/3/library/venv.html) is recommended.

### Setup

```bash
git clone https://github.com/Denko-AI/colosseum-.git
cd colosseum-
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and add your API key(s). You only need keys for the providers you plan to use:

```
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
MISTRAL_API_KEY="..."
```

See `.env.example` for the full list of supported providers.

### Run

The easiest way to start is the interactive launcher:

```bash
make interactive
```

This walks you through picking models, mode (text or vision), temperature, and game options — then starts the match with a live TUI dashboard.

---

## Other Ways to Run

| Command | What it does |
|---|---|
| `make interactive` | Interactive setup menu → live dashboard |
| `make run` | Quick match with defaults (edit `script.py` to configure) |
| `make local` | Local models via [Ollama](https://ollama.com/) (edit `local.py`) |
| `make demo` | Demo match → displays win screen |
| `make go` | Continuous matches in a loop |

### Using Ollama (local models)

1. Install and start Ollama: `ollama serve`
2. Pull a model: `ollama pull mistral`
3. Run: `make local`

Models use the format `ollama:<model_name>`. Edit `local.py` to change which models fight.

### Docker

```bash
# Build
docker build -t diambra-app .

# Run
docker run --name diambra-container -v ~/.diambra/roms:/app/roms diambra-app

# Or with Docker Compose (includes Ollama)
docker-compose up
```

---

## How It Works

Each player is controlled by an LLM that receives game state and responds with a list of moves in real time. Two modes are available:

**Text mode** — the LLM receives a text description of positions, health, super meter, and recent actions. It reasons about distance and strategy to pick moves.

**Vision mode** — the LLM receives a screenshot of the current game frame and decides moves based purely on visual information.

Both modes run in separate threads with real-time streaming. The TUI dashboard shows health bars, super meters, move logs, and LLM thinking indicators as the fight plays out.

![fight3 drawio](https://github.com/OpenGenerativeAI/llm-colosseum/assets/78322686/3a212601-f54c-490d-aeb9-6f7c2401ebe6)

---

## Leaderboard

546 fights across 14 models. Full rankings on [Hugging Face](https://huggingface.co/spaces/junior-labs/llm-colosseum).

| Rank | Model | ELO |
|---:|:---|---:|
| 1 | 🥇 openai:gpt-4o (text) | 1912 |
| 2 | 🥈 openai:gpt-4o-mini (vision) | 1835 |
| 3 | 🥉 openai:gpt-4o-mini (text) | 1671 |
| 4 | openai:gpt-4o (vision) | 1657 |
| 5 | mistral:pixtral-large-latest (vision) | 1655 |
| 6 | mistral:pixtral-12b-2409 (vision) | 1591 |
| 7 | mistral:pixtral-12b-2409 (text) | 1569 |

<details>
<summary>Full rankings (14 models)</summary>

| Rank | Model | ELO |
|---:|:---|---:|
| 8 | together:meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo (text) | 1441 |
| 9 | anthropic:claude-3-haiku-20240307 (vision) | 1365 |
| 10 | mistral:pixtral-large-latest (text) | 1356 |
| 11 | anthropic:claude-3-haiku-20240307 (text) | 1334 |
| 12 | anthropic:claude-3-sonnet-20240229 (vision) | 1315 |
| 13 | together:meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo (vision) | 1270 |
| 14 | anthropic:claude-3-sonnet-20240229 (text) | 1029 |

*Claude 3 Sonnet scored low due to frequent refusals to fight and high API latency.*

</details>

![Win rate matrix](notebooks/result_matrix.png)

---

## Customization

### Change the prompts

The LLM prompt lives in `agent/robot.py` inside `TextRobot.call_llm()` and `VisionRobot.call_llm()`. Edit these to change fighting strategy, personality, or move selection logic.

### Add a new model provider

1. Add the provider and model to `agent/config.py` in the `MODELS` dict
2. Add the LlamaIndex integration to `agent/llm.py`
3. Add the API key env var to `.env.example` and `Player._PROVIDER_API_KEYS` in `eval/game.py`

### Submit your model to the leaderboard

Create a class that inherits from `Robot` with your custom logic and open a PR.

---

## Test Mode

Run without LLM API calls (moves are chosen randomly):

```bash
# Set in .env
DISABLE_LLM="True"
```

---

## Project Structure

```
colosseum/
├── main.py              # Interactive launcher with TUI setup
├── script.py            # Quick-run entry point
├── local.py             # Ollama entry point
├── demo.py              # Demo entry point
├── agent/
│   ├── config.py        # Models, moves, combos, meta-instructions
│   ├── llm.py           # LlamaIndex client factory
│   ├── observer.py      # Color-based position detection
│   └── robot.py         # TextRobot & VisionRobot (LLM prompts live here)
├── eval/
│   ├── game.py          # Game loop, player threads, DIAMBRA integration
│   └── display.py       # Rich TUI dashboard & post-game summary
└── tests/               # Test suite (runs without DIAMBRA or API keys)
```

---

## Credits

Made with ❤️ by the OpenGenerativeAI team from [phospho](https://phospho.ai) (@oulianov @Pierre-LouisBJT @Platinn) and [Quivr](https://www.quivr.app) (@StanGirard) during Mistral Hackathon 2024 in San Francisco.
