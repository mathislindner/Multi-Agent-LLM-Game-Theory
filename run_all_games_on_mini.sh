#!/bin/bash

games=("stag_hunt" "chicken" "hawk_dove", "coordination") #"prisoners_dilemma" "generic"  "deadlock" "battle_of_sexes)
personas=("NONE" "ISTJ" "ISFJ" "INFJ" "INTJ" "ISTP" "ISFP" "INFP" "INTP" "ESTP" "ESFP" "ENFP" "ENTP" "ESTJ" "ESFJ" "ENFJ" "ENTJ" "ALTRUISTIC" "SELFISH")
nr_of_rounds=7
model_id_1="gpt-4o-mini-2024-07-18"
model_id_2="gpt-4o-mini-2024-07-18"

total_iterations=$(( (${#games[@]} * ${#personas[@]} * (${#personas[@]} + 1)) / 2 ))
current_iteration=0

for game_name in "${games[@]}"; do
    for i in "${!personas[@]}"; do
        for ((j=i; j<${#personas[@]}; j++)); do  # Start from i, allowing (A, A)
            agent_1="${personas[$i]}"
            agent_2="${personas[$j]}"
            current_iteration=$((current_iteration + 1))
            echo "Iteration $current_iteration of $total_iterations: $agent_1 vs $agent_2 in $game_name"
            python main.py --model_id_1 $model_id_1 --model_id_2 $model_id_2 --rounds $nr_of_rounds --agent_1_persona "$agent_1" --agent_2_persona "$agent_2" --game_name $game_name
        done
    done
done
