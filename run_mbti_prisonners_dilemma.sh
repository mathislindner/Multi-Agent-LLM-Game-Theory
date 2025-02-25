#!/bin/bash

personas=("ISTJ" "ISFJ" "INFJ" "INTJ" "ISTP" "ISFP" "INFP" "INTP" "ESTP" "ESFP" "ENFP" "ENTP" "ESTJ" "ESFJ" "ENFJ" "ENTJ" "EXPERT" "ALTRUISTIC" "SELFISH" "NONE" )
nr_of_rounds=20
model_id="gpt-4o-mini"

for i in "${!personas[@]}"; do
    for j in $(seq $((i + 1)) ${#personas[@]}); do
        agent_1="${personas[$i]}"
        agent_2="${personas[$j]}"
        python main.py --model_id $model_id --rounds $nr_of_rounds --agent_1_persona "$agent_1" --agent_2_persona "$agent_2" --game_name prisoners_dilemma
    done
done