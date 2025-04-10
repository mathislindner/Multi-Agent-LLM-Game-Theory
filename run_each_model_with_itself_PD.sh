#!/bin/bash

personas=("NONE" "ISTJ" "ISFJ" "INFJ" "INTJ" "ISTP" "ISFP" "INFP" "INTP" "ESTP" "ESFP" "ENFP" "ENTP" "ESTJ" "ESFJ" "ENFJ" "ENTJ" "ALTRUISTIC" "SELFISH")
nr_of_rounds=7
model_providers=("anthropic" "anthropic" "google_vertexai" "google_vertexai" "deepseek")

model_ids=("claude-3-5-haiku-20241022" "claude-3-5-sonnet-20241022" "gemini-2.0-flash" "gemini-2.0-flash-lite" "deepseek-chat")

game_name="prisoners_dilemma"

total_iterations=$(( (${#personas[@]} * (${#personas[@]} + 1)) / 2 * ${#model_ids[@]} ))
echo "Total iterations: $total_iterations"
current_iteration=0
exit
for m in "${!model_ids[@]}"; do
    model_id="${model_ids[$m]}"
    provider="${model_providers[$m]}"
    for i in "${!personas[@]}"; do
        for ((j=i; j<${#personas[@]}; j++)); do  # Start from i, allowing (A, A)
            agent_1="${personas[$i]}"
            agent_2="${personas[$j]}"
            current_iteration=$((current_iteration + 1))
            echo "Iteration $current_iteration of $((total_iterations * ${#model_ids[@]})): $agent_1 vs $agent_2 with $model_id ($provider)"
            python main.py --model_id_1 $model_id --model_id_2 $model_id --model_provider_1 $provider --model_provider_2 $provider --rounds $nr_of_rounds --agent_1_persona "$agent_1" --agent_2_persona "$agent_2" --game_name $game_name
        done
    done
done
