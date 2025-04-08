#!/bin/bash

personas=("NONE" "ISTJ" "ISFJ" "INFJ" "INTJ" "ISTP" "ISFP" "INFP" "INTP" "ESTP" "ESFP" "ENFP" "ENTP" "ESTJ" "ESFJ" "ENFJ" "ENTJ" "ALTRUISTIC" "SELFISH")
nr_of_rounds=7
model_ids=("deepseek-chat","gpt-4o-mini-2024-07-18" "claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219")
game_name="generic"

total_iterations=$(( (${#personas[@]} * (${#personas[@]} + 1)) / 2 ))
current_iteration=0

for m in "${!model_ids[@]}"; do
    for n in "${!model_ids[@]}"; do
        if (( m <= n )); then  # Ensure each model pair (including itself) is only played once
            model_id_1="${model_ids[$m]}"
            model_id_2="${model_ids[$n]}"
            for i in "${!personas[@]}"; do
                for ((j=i; j<${#personas[@]}; j++)); do  # Start from i, allowing (A, A)
                    agent_1="${personas[$i]}"
                    agent_2="${personas[$j]}"
                    current_iteration=$((current_iteration + 1))
                    echo "Iteration $current_iteration of $((total_iterations * ${#model_ids[@]} * (${#model_ids[@]} + 1) / 2)): $agent_1 vs $agent_2 with $model_id_1 vs $model_id_2"
                    python main.py --model_id_1 $model_id_1 --model_id_2 $model_id_2 --rounds $nr_of_rounds --agent_1_persona "$agent_1" --agent_2_persona "$agent_2" --game_name $game_name
                done
            done
        fi
    done
done
