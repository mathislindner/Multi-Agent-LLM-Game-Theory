# LLM Agents playing Game Theory Games

This project simulates game rounds between two AI agents with distinct personalities based on MBTI profiles. The agents play strategic games such as the Prisoner's Dilemma, Stag Hunt, Chicken, and others. The results are saved as CSV files for analysis.

---

## Overview

The main script runs a specified number of rounds of a given game between two AI agents. Each agent is assigned a personality profile (from MBTI types) and uses a language model (from specified providers) to make decisions during the game.

---

## How It Works

- The script imports the game logic from `run_n_rounds_w_com` in the `src.run_games_mbti` module.
- It generates an output file named with the current date in `src/data/outputs/`.
- It supports multiple game types:
  - `prisoners_dilemma`
  - `stag_hunt`
  - `generic`
  - `chicken`
  - `coordination`
  - `hawk_dove`
  - `deadlock`
  - `battle_of_sexes`
- Agent personalities are chosen from predefined MBTI personality keys loaded from a JSON prompt file.
- Models can be specified along with optional providers.

---

## Usage

Run the script with the following command-line arguments:

```bash
python main.py \
  --model_id_1 MODEL_ID_1 \
  --model_id_2 MODEL_ID_2 \
  --model_provider_1 PROVIDER_1 \
  --model_provider_2 PROVIDER_2 \
  --rounds NUMBER_OF_ROUNDS \
  --agent_1_persona AGENT_1_PERSONALITY \
  --agent_2_persona AGENT_2_PERSONALITY \
  --game_name GAME_NAME
```
## Arguments
| Argument             | Type   | Description                              | Required | Options                                                                                                            |
| -------------------- | ------ | ---------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------ |
| `--model_id_1`       | string | ID of the first model to use             | Yes      | Any valid model ID                                                                                                 |
| `--model_id_2`       | string | ID of the second model to use            | Yes      | Any valid model ID                                                                                                 |
| `--model_provider_1` | string | Provider for the first model (optional)  | No       | e.g., `openai`, `anthropic`                                                                                        |
| `--model_provider_2` | string | Provider for the second model (optional) | No       | e.g., `openai`, `anthropic`                                                                                        |
| `--rounds`           | int    | Number of rounds to simulate             | Yes      | Positive integer                                                                                                   |
| `--agent_1_persona`  | string | Personality key for agent 1              | Yes      | From MBTI keys in `mbti_prompts_250129.json`                                                                       |
| `--agent_2_persona`  | string | Personality key for agent 2              | Yes      | From MBTI keys in `mbti_prompts_250129.json`                                                                       |
| `--game_name`        | string | Game type to play                        | No       | `prisoners_dilemma`, `stag_hunt`, `generic`, `chicken`, `coordination`, `hawk_dove`, `deadlock`, `battle_of_sexes` |


## Output
- Results are saved as CSV files in the src/data/outputs/ directory.
- File names follow the pattern: YYMMDD.csv where YYMMDD is the current date.
## Example

```bash
python your_script.py \
  --model_id_1 "gpt-4" \
  --model_id_2 "gpt-3.5-turbo" \
  --rounds 10 \
  --agent_1_persona "INTJ" \
  --agent_2_persona "ENFP" \
  --game_name "prisoners_dilemma"
```
