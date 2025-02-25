#!/bin/bash

personas=("NONE" "ISTJ" "ISFJ" "INFJ" "INTJ" "ISTP" "ISFP" "INFP" "INTP" "ESTP" "ESFP" "ENFP" "ENTP" "ESTJ" "ESFJ" "ENFJ" "ENTJ" "ALTRUISTIC" "SELFISH")
nr_of_rounds=7
model_id="gpt-4o-mini"
game_name="prisoners_dilemma"

total_iterations=$((${#personas[@]} * ${#personas[@]}))
current_iteration=0

for i in "${!personas[@]}"; do
    for j in "${!personas[@]}"; do
        agent_1="${personas[$i]}"
        agent_2="${personas[$j]}"
        current_iteration=$((current_iteration + 1))
        echo "Iteration $current_iteration of $total_iterations: $agent_1 vs $agent_2"
        python main.py --model_id $model_id --rounds $nr_of_rounds --agent_1_persona "$agent_1" --agent_2_persona "$agent_2" --game_name $game_name
    done
done