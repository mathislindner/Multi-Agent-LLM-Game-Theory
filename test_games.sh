#!/bin/bash

games=("prisoners_dilemma" "stag_hunt")
persona="INFJ"  # Randomly chosen persona for testing
nr_of_rounds=2
model_id_1="gpt-4o-mini-2024-07-18"
model_id_2="gpt-4o-mini-2024-07-18"

total_iterations=${#games[@]}
current_iteration=0

for game_name in "${games[@]}"; do
    current_iteration=$((current_iteration + 1))
    echo "Iteration $current_iteration of $total_iterations: Testing $persona in $game_name"
    python main.py --model_id_1 $model_id_1 --model_id_2 $model_id_2 --rounds $nr_of_rounds --agent_1_persona "$persona" --agent_2_persona "$persona" --game_name $game_name
done
