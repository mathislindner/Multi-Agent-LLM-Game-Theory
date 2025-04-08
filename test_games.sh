#!/bin/bash

games=("prisoners_dilemma")
persona="ENFJ"  # Randomly chosen persona for testing
nr_of_rounds=2
#model_id_1="gpt-4o-mini-2024-07-18"
#model_id_2="gpt-4o-mini-2024-07-18"
#model_id_1="deepseek-chat"
#model_id_2="deepseek-chat"
#model_provider_1="google_genai"
#model_provider_2="google_genai"
#model_id_1="gemini-1.5-pro-001"
#model_id_2="gemini-1.5-pro-001"

#model_provider_1="google_genai"
#model_provider_2="google_genai"
model_provider_1="anthropic"
model_provider_2="anthropic"
model_id_1="claude-3-5-haiku-20241022"
model_id_2="claude-3-5-haiku-20241022"
total_iterations=${#games[@]}
current_iteration=0

for game_name in "${games[@]}"; do
    current_iteration=$((current_iteration + 1))
    echo "Iteration $current_iteration of $total_iterations: Testing $persona in $game_name"
    #python main.py --model_provider_1 $model_provider_1 --model_provider_2 $model_provider_2 --model_id_1 $model_id_1 --model_id_2 $model_id_2 --rounds $nr_of_rounds --agent_1_persona "$persona" --agent_2_persona "$persona" --game_name $game_name
    python main.py --model_id_1 $model_id_1 --model_id_2 $model_id_2 --rounds $nr_of_rounds --agent_1_persona "$persona" --agent_2_persona "$persona" --game_name $game_name

done

