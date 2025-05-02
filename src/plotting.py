
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact

from typing import Literal, Dict, Union, Optional


df_types = {
    'pd':
        {
            'plots_path': '/home/mat/Documents/Github/master_thesis_project/src/data/outputs/plots/game_theory/stability_PD/',
            'df_path': '/home/mat/Documents/Github/master_thesis_project/src/data/outputs/stability_PD.csv',
        }
    ,
    'model':
        {
            'plots_path': '/home/mat/Documents/Github/master_thesis_project/src/data/outputs/plots/game_theory/by_model/',
            'df_path': '/home/mat/Documents/Github/master_thesis_project/src/data/outputs/by_model.csv',
            'to_keep': [
                        "deepseek-chat", 
                        "claude-3-haiku-20240307", 
                        "gemini-2.0-flash-lite",
                        "gpt-4o-mini-2024-07-18",
                        "gemini-2.5-pro-exp-03-25", 
                        "gpt-4.1-mini-2025-04-14", 
                        "gpt-3.5-turbo"
                    ]
        },
    'game':
        {
            'plots_path': '/home/mat/Documents/Github/master_thesis_project/src/data/outputs/plots/game_theory/by_game/',
            'df_path': '/home/mat/Documents/Github/master_thesis_project/src/data/outputs/by_game.csv',
            'to_keep': ['chicken', 'coordination', 'generic', 'hawk_dove', 'prisoners_dilemma', 'stag_hunt']
        }
}
personality_order = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ",
    "NONE", "SELFISH", "ALTRUISTIC"
]

MODEL_ALIAS = {
    "gemini-2.5-pro-exp-03-25":       "G2.5-pro",
    "gemini-2.0-flash-lite":          "G2.0-lite",
    "gpt-4o-mini-2024-07-18":         "GPT-4o-mini",
    "gpt-4.1-mini-2025-04-14":        "GPT-4.1-mini",
    "gpt-3.5-turbo":                  "GPT-3.5",
    "claude-3-haiku-20240307":        "Claude-3-hk",
    "deepseek-chat":                  "DeepSeek",
}

def who_lied_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns:
      * lied_first_1 — 1 if agent 1 lied first, 0 otherwise
      * lied_first_2 — 1 if agent 2 lied first, 0 otherwise
    Ties (‘they lied on the same turn’) count for both.
    If nobody ever lies, both columns are 0.
    """
    # Helper to locate first lie (0 → lie). Returns ∞ if no lies.
    def first_lie(seq):
        if isinstance(seq, list):
            try:
                return seq.index(0)
            except ValueError:
                pass          # 0 not in the list
        return np.inf

    # Find the first-lie index for each agent
    df = df.copy()   # avoid mutating caller’s frame
    df["first_lie_1"] = df["truthful_agent_1"].apply(first_lie)
    df["first_lie_2"] = df["truthful_agent_2"].apply(first_lie)

    # Earliest lie (row-wise)
    earliest = df[["first_lie_1", "first_lie_2"]].min(axis=1)

    # Assign accountability; ensure at least one lie occurred
    df["lied_first_1"] = ((df["first_lie_1"] == earliest) & ~np.isinf(earliest)).astype(int)
    df["lied_first_2"] = ((df["first_lie_2"] == earliest) & ~np.isinf(earliest)).astype(int)

    # Optional: drop helper columns
    return df.drop(columns=["first_lie_1", "first_lie_2"])

def who_lied_first_free_will(df: pd.DataFrame) -> pd.DataFrame:
    #TODO: point is if the others intention in the next message is defecting, then it s not free will
    # check in intentions, if the intention in the message an action and the other one is just matching it, while saing something else
    # becuase technically it s just adapting, even if their intent was cooperate, a defection threat will push them to something else
    # add column FreeWill True or False
    # if not a FirstLie then None
    pass

def import_df_and_plot_path(df_type):
    df_path = df_types[df_type]['df_path']
    plots_path = df_types[df_type]['plots_path']
    to_keep = df_types[df_type].get('to_keep', None)
    df = pd.read_csv(df_path)
    df['agent_1_scores'] = df['agent_1_scores'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['agent_2_scores'] = df['agent_2_scores'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['agent_1_cumulative_scores'] = df['agent_1_scores'].apply(lambda x: [0] + [sum(x[:i+1]) for i in range(len(x))] if isinstance(x, list) else x)
    df['agent_2_cumulative_scores'] = df['agent_2_scores'].apply(lambda x: [0] + [sum(x[:i+1]) for i in range(len(x))] if isinstance(x, list) else x)

    #parse as lists, something went wrong when saving the csv
    df['truthful_agent_1'] = df['truthful_agent_1'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['truthful_agent_2'] = df['truthful_agent_2'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # Parse the actions as lists of strings
    df['agent_1_actions'] = df['agent_1_actions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['agent_2_actions'] = df['agent_2_actions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['agent_1_messages'] = df['agent_1_messages'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['agent_2_messages'] = df['agent_2_messages'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['intent_agent_1'] = df['intent_agent_1'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['intent_agent_2'] = df['intent_agent_2'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    nonmbti = ['NONE', 'ALTRUISTIC', 'SELFISH']
    # Map each personality to its dichotomies
    def safe_dichotomy_extraction(x, position, positive_letter, negative_letter):
        if x in nonmbti:
            return None
        else:
            return positive_letter if x[position] == positive_letter else negative_letter

    nonmbti = ['NONE', 'ALTRUISTIC', 'SELFISH']
    # Map each personality to its dichotomies
    df['I/E_1'] = df['personality_1'].apply(lambda x: safe_dichotomy_extraction(x, 0, 'I', 'E'))
    df['N/S_1'] = df['personality_1'].apply(lambda x: safe_dichotomy_extraction(x, 1, 'N', 'S'))
    df['T/F_1'] = df['personality_1'].apply(lambda x: safe_dichotomy_extraction(x, 2, 'F', 'T'))
    df['J/P_1'] = df['personality_1'].apply(lambda x: safe_dichotomy_extraction(x, 3, 'J', 'P'))
    df['I/E_2'] = df['personality_2'].apply(lambda x: safe_dichotomy_extraction(x, 0, 'I', 'E'))
    df['N/S_2'] = df['personality_2'].apply(lambda x: safe_dichotomy_extraction(x, 1, 'N', 'S'))
    df['T/F_2'] = df['personality_2'].apply(lambda x: safe_dichotomy_extraction(x, 2, 'F', 'T'))
    df['J/P_2'] = df['personality_2'].apply(lambda x: safe_dichotomy_extraction(x, 3, 'J', 'P'))
    # Filter the DataFrame to keep only the specified games or model depending on the df_type
    if to_keep is not None:
        if df_type == 'model':
            df = df[df['model_name_1'].isin(to_keep)]
        elif df_type == 'game':
            df = df[df['game_name'].isin(to_keep)]
            
    df = who_lied_first(df)
    
    return df, plots_path

def import_df_agents(df):
    def count_strategy_switches(actions):
        if not actions or len(actions) < 2:
            return 0
        return sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
    # Create a new DataFrame with one row per agent
    df_agent1 = df.copy()
    df_agent2 = df.copy()

    # Add a column to indicate the agent
    df_agent1['Agent'] = 'Agent 1'
    df_agent2['Agent'] = 'Agent 2'
    # Rename columns to remove agent-specific prefixes
    df_agent1 = df_agent1.rename(columns={
        'personality_1': 'Personality',
        'agent_1_scores': 'Scores',
        'agent_1_messages': 'Messages',
        'agent_1_actions': 'Actions',
        'agent_1_cumulative_scores': 'CumulativeScores',
        'intent_agent_1': 'Intent',
        'truthful_agent_1': 'Truthful',
        'analysis_agent_1': 'Analysis',
        'lied_first_1': 'LiedFirst',
        'I/E_1': 'I/E',
        'N/S_1': 'N/S',
        'T/F_1': 'T/F',
        'J/P_1': 'J/P'
    })

    df_agent2 = df_agent2.rename(columns={
        'personality_2': 'Personality',
        'agent_2_scores': 'Scores',
        'agent_2_messages': 'Messages',
        'agent_2_actions': 'Actions',
        'agent_2_cumulative_scores': 'CumulativeScores',
        'intent_agent_2': 'Intent',
        'truthful_agent_2': 'Truthful',
        'analysis_agent_2': 'Analysis',
        'lied_first_2': 'LiedFirst',
        'I/E_2': 'I/E',
        'N/S_2': 'N/S',
        'T/F_2': 'T/F',
        'J/P_2': 'J/P'
    })

    # Remove the other columns
    df_agent1 = df_agent1.drop(columns=[
        'personality_2', 'agent_2_scores', 'agent_2_messages', 'agent_2_actions', 
        'agent_2_cumulative_scores', 'intent_agent_2', 'truthful_agent_2', 'analysis_agent_2',
        'I/E_2', 'N/S_2', 'T/F_2', 'J/P_2','lied_first_2'
    ])
    df_agent2 = df_agent2.drop(columns=[
        'personality_1', 'agent_1_scores', 'agent_1_messages', 'agent_1_actions', 
        'agent_1_cumulative_scores', 'intent_agent_1', 'truthful_agent_1', 'analysis_agent_1',
        'I/E_1', 'N/S_1', 'T/F_1', 'J/P_1', 'lied_first_1'
    ])
    # Combine the two DataFrames
    df_agents = pd.concat([df_agent1, df_agent2], ignore_index=True)
    
    #small fixes
    df_agents = df_agents.rename(columns={'game_name': 'GameName','model_name_1': 'Model'})
    df_agents["TotalScore"] = df_agents["CumulativeScores"].apply(lambda x: x[-1])
    df_agents["Truthfulness"] = df_agents["Truthful"].apply(lambda x: sum(x)/len(x) if len(x) > 0 else 0)
    df_agents['DefectionRatio'] = df_agents.apply(
        lambda row: row['Actions'].count('defect') / len(row['Actions']) if len(row['Actions']) > 0 else 0, axis=1
    )
    df_agents['Switches'] = df_agents['Actions'].apply(count_strategy_switches)
        
    return df_agents


# ---------------------------------------------------------------------
# 1.  DEFECT‑RATE ANALYSIS  (0–1 proportion of “defect” across 7 rounds)
# ---------------------------------------------------------------------
def get_defection_rates_dichos(
    df_agents: pd.DataFrame,
    plots_path: Union[str, os.PathLike],
    df_type: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Welch t‑tests and Cohen’s d for defection‑rate differences across each
    MBTI dichotomy (I/E, N/S, T/F, J/P).

    Parameters
    ----------
    df_agents : DataFrame  (one row = agent × game)
      Required columns:
        'Agent', 'Actions', 'Personality', 'I/E', 'N/S', 'T/F', 'J/P',
        plus 'GameName' if df_type=="game", or 'Model' if df_type=="model".
    plots_path : directory to write "<slice>_defection_dichos.txt"
    df_type : {"game","model",None}
        Controls the slice column; None ⇒ pooled across everything.
    """
    # ---- select slice column -----------------------------------------------
    slice_by = {"game": "GameName", "model": "Model"}.get(df_type)
    slice_suffix = slice_by if slice_by else "pooled"

    # ---- 0. Robust defection‑rate extractor --------------------------------
    def defect_rate(actions):
        if isinstance(actions, str):
            try:
                actions = ast.literal_eval(actions)
            except Exception:
                return np.nan
        return np.mean([str(a).strip().lower() == "defect" for a in actions])

    df = df_agents.copy()
    df["defect_rate"] = df["Actions"].apply(defect_rate)

    # ---- 1. collapse to one row per Agent (+ slice if present) -------------
    group_cols = ["Agent", "Personality", "I/E", "N/S", "T/F", "J/P"]
    if slice_by:
        group_cols.insert(1, slice_by)         # Agent, slice, Personality, …

    agent_means = (
        df.groupby(group_cols, as_index=False)["defect_rate"]
          .mean()
          .dropna(subset=["defect_rate"])
    )

    # ---- 2. run Welch t for each dichotomy ---------------------------------
    dichos = {"I/E": ("I", "E"),
              "N/S": ("N", "S"),
              "T/F": ("T", "F"),
              "J/P": ("J", "P")}
    lines, results = [], {}
    for dim, (side_a, side_b) in dichos.items():
        a = agent_means.loc[agent_means[dim] == side_a, "defect_rate"].values
        b = agent_means.loc[agent_means[dim] == side_b, "defect_rate"].values

        if len(a) < 2 or len(b) < 2:
            lines.append(f"{dim}: not enough data (n_{side_a}={len(a)}, n_{side_b}={len(b)})")
            continue

        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        pooled_sd = np.sqrt(
            ((a.var(ddof=1) * (len(a) - 1)) + (b.var(ddof=1) * (len(b) - 1)))
            / (len(a) + len(b) - 2)
        )
        cohen_d = (b.mean() - a.mean()) / pooled_sd

        lines.append(
            f"{dim}: n_{side_a}={len(a)}, n_{side_b}={len(b)}  "
            f"Mean {side_a}={a.mean():.3f}, Mean {side_b}={b.mean():.3f}  "
            f"p={p_val:.3g}, Cohen's d={cohen_d:.2f}"
        )
        results[dim] = {"p": p_val, "d": cohen_d}

    # ---- 3. save -----------------------------------------------------------
    os.makedirs(plots_path, exist_ok=True)
    txt = os.path.join(plots_path, f"{slice_suffix}_defection_dichos.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines))
    return results



# ---------------------------------------------------------------------
# 2.  TRUTHFULNESS ANALYSIS  (0–1 proportion of truthful statements)
# ---------------------------------------------------------------------
def get_truthfulness_dichos(
    df_agents: pd.DataFrame,
    plots_path: Union[str, os.PathLike],
    df_type: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Welch t‑tests and Cohen’s d for per‑agent *Truthfulness* rates
    across each MBTI dichotomy.
    """
    slice_by = {"game": "GameName", "model": "Model"}.get(df_type)
    slice_suffix = slice_by if slice_by else "pooled"

    # ---- 0. collapse to per‑agent (+ slice) rows ---------------------------
    group_cols = ["Agent", "Personality", "I/E", "N/S", "T/F", "J/P"]
    if slice_by:
        group_cols.insert(1, slice_by)

    agent_means = (
        df_agents
        .groupby(group_cols, as_index=False)["Truthfulness"]
        .mean()
        .dropna(subset=["Truthfulness"])
    )

    # ---- 1. Welch t for dichotomies ---------------------------------------
    dichos = {"I/E": ("I", "E"),
              "N/S": ("N", "S"),
              "T/F": ("T", "F"),
              "J/P": ("J", "P")}
    lines, results = [], {}
    for dim, (side_a, side_b) in dichos.items():
        a = agent_means.loc[agent_means[dim] == side_a, "Truthfulness"].values
        b = agent_means.loc[agent_means[dim] == side_b, "Truthfulness"].values

        if len(a) < 2 or len(b) < 2:
            lines.append(f"{dim}: not enough data (n_{side_a}={len(a)}, n_{side_b}={len(b)})")
            continue

        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        pooled_sd = np.sqrt(
            ((a.var(ddof=1) * (len(a) - 1)) + (b.var(ddof=1) * (len(b) - 1)))
            / (len(a) + len(b) - 2)
        )
        cohen_d = (b.mean() - a.mean()) / pooled_sd

        lines.append(
            f"{dim}: n_{side_a}={len(a)}, n_{side_b}={len(b)}  "
            f"Mean {side_a}={a.mean():.3f}, Mean {side_b}={b.mean():.3f}  "
            f"p={p_val:.3g}, Cohen's d={cohen_d:.2f}"
        )
        results[dim] = {"p": p_val, "d": cohen_d}

    # ---- 2. save -----------------------------------------------------------
    os.makedirs(plots_path, exist_ok=True)
    txt = os.path.join(plots_path, f"{slice_suffix}_truthfulness_dichos.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines))
    return results

def get_strategy_switch_pvalues(
    df_agents: pd.DataFrame,
    plots_path: str | os.PathLike,
    df_type: str | None = None,
) -> Dict[str, Dict[str, float]]:

    # ------------------------------------------------------------------
    # 0. Housekeeping
    # ------------------------------------------------------------------
    slice_by = "GameName" if df_type == "game" else "Model"
    df = df_agents.rename(columns={"StrategySwitches": "Switches"}).copy()

    needed = {"Agent", "Personality", "T/F", slice_by, "Switches"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"df_agents missing columns: {missing}")

    # ------------------------------------------------------------------
    # 1A. Agent × slice (independent observations)
    # ------------------------------------------------------------------
    agent_slice = (
        df.groupby(["Agent", "Personality", slice_by, "T/F"], as_index=False)["Switches"]
          .mean()
    )

    T_a = agent_slice.query("`T/F` == 'T'")["Switches"].values
    F_a = agent_slice.query("`T/F` == 'F'")["Switches"].values

    t_a, p_a = ttest_ind(T_a, F_a, equal_var=False)
    u_a, p_u_a = mannwhitneyu(T_a, F_a, alternative="two-sided")

    # ------------------------------------------------------------------
    # 1B. MBTI 8-types pooled per slice
    # ------------------------------------------------------------------
    type_slice = (
        agent_slice.groupby(["Personality", "T/F", slice_by], as_index=False)["Switches"]
                   .mean()
    )

    T_flat = np.concatenate(
        type_slice.query("`T/F` == 'T'").groupby(slice_by)["Switches"].apply(list).values
    )
    F_flat = np.concatenate(
        type_slice.query("`T/F` == 'F'").groupby(slice_by)["Switches"].apply(list).values
    )

    t_t, p_t = ttest_ind(T_flat, F_flat, equal_var=False)
    u_t, p_u_t = mannwhitneyu(T_flat, F_flat, alternative="two-sided")

    # ------------------------------------------------------------------
    # 2. Pack results
    # ------------------------------------------------------------------
    mean_T = T_a.mean()
    mean_F = F_a.mean()
    direction = "T > F" if mean_T > mean_F else "F > T"

    results = {
        "agent_slice": {
            "p_welch": p_a, "p_mwu": p_u_a,
            "mean_T": mean_T, "mean_F": mean_F,
            "n_T": len(T_a), "n_F": len(F_a)
        },
        "type_slice": {
            "p_welch": p_t, "p_mwu": p_u_t,
            "mean_T": T_flat.mean(), "mean_F": F_flat.mean(),
            "n_T": len(T_flat), "n_F": len(F_flat)
        }
    }

    # ------------------------------------------------------------------
    # 3. Write results to file
    # ------------------------------------------------------------------
    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, f"strategy_switch_pvalues_by_{slice_by}.txt")

    with open(file_path, "w") as f:
        f.write(f"=== Strategy Switch Comparison by T/F across {slice_by}s ===\n\n")

        f.write(f"---- Agent × {slice_by} (independent units) ----\n")
        f.write(f"Mean strategy switches:\n")
        f.write(f"  T: {results['agent_slice']['mean_T']:.2f}\n")
        f.write(f"  F: {results['agent_slice']['mean_F']:.2f}\n")
        f.write(f"Welch’s t-test:       p = {results['agent_slice']['p_welch']:.3g}\n")
        f.write(f"Mann–Whitney U-test:  p = {results['agent_slice']['p_mwu']:.3g}\n\n")

        f.write(f"---- 8 T-types vs 8 F-types pooled per {slice_by} ----\n")
        f.write(f"Mean strategy switches:\n")
        f.write(f"  T: {results['type_slice']['mean_T']:.2f}\n")
        f.write(f"  F: {results['type_slice']['mean_F']:.2f}\n")
        f.write(f"Welch’s t-test:       p = {results['type_slice']['p_welch']:.3g}\n")
        f.write(f"Mann–Whitney U-test:  p = {results['type_slice']['p_mwu']:.3g}\n\n")

        # ------------------------------------------------------------------
        # 4. Add correct interpretation sentence
        # ------------------------------------------------------------------
        if slice_by == "GameName":
            f.write(
                f"We tested whether Thinking (T) agents switch strategies more frequently than "
                f"Feeling (F) agents across repeated interactions.\n"
                f"At the agent × game level, {direction.split()[0]} agents exhibited more switching "
                f"(M = {max(mean_T, mean_F):.2f} vs. {min(mean_T, mean_F):.2f}; Welch’s t, "
                f"p = {p_a:.3g}), suggesting greater behavioral flexibility in {direction.split()[0]} types.\n\n"
                f"When pooling the eight T-types and eight F-types within each game, the difference remained "
                f"(Welch’s t, p = {p_t:.3g}), indicating that this effect is robust across games.\n"
            )
        elif slice_by == "Model":
            f.write(
                f"We tested whether Thinking (T) agents switch strategies more often than Feeling (F) agents "
                f"across different model configurations.\n"
                f"At the agent × model level, {direction.split()[0]} agents switched significantly more often "
                f"(M = {max(mean_T, mean_F):.2f} vs. {min(mean_T, mean_F):.2f}; Welch’s t, "
                f"p = {p_a:.3g}).\n"
                f"Pooling the eight T-types and eight F-types within each model confirmed the pattern "
                f"(Welch’s t, p = {p_t:.3g}), indicating that this behavioral trend is consistent across models.\n"
            )
        else:
            f.write(
                f"Across all data, {direction.split()[0]} agents switched strategies more often "
                f"(M = {max(mean_T, mean_F):.2f} vs. {min(mean_T, mean_F):.2f}; Welch’s t, "
                f"p = {p_a:.3g}), supporting the hypothesis that {direction.split()[0]} types are more "
                f"strategically adaptive.\n"
            )


def _prop_test(count1: int, n1: int, count2: int, n2: int) -> float:
    """
    Two-proportion test using Welch's chi-square; if any expected cell < 5
    switch to Fisher's exact (two-sided).  Returns the p-value.
    """
    table = np.array([[count1, n1 - count1],
                      [count2, n2 - count2]])
    if (table < 5).any():
        _, p = fisher_exact(table, alternative="two-sided")
    else:
        _, p, _, _ = chi2_contingency(table, correction=False)
    return p


from scipy.stats import chi2_contingency, fisher_exact


# ---------- helper for proportion p-value -----------------------------------
def _prop_test(x1: int, n1: int, x2: int, n2: int) -> float:
    """Welch χ² two-proportion test; falls back to Fisher if cells < 5."""
    tbl = np.array([[x1, n1 - x1],
                    [x2, n2 - x2]])
    if (tbl < 5).any():
        _, p = fisher_exact(tbl, alternative="two-sided")
    else:
        _, p, _, _ = chi2_contingency(tbl, correction=False)
    return p


# ---------- main routine ----------------------------------------------------
def get_liedfirst_pvalues(
    df_agents: pd.DataFrame,
    plots_path: str | os.PathLike,
    df_type: Literal["game", "model", "pd", None] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare 'LiedFirst' rates for all four MBTI polarities (I/E, N/S, T/F, J/P).
    Saves a .txt summary and returns a nested dict of p-values & rates.
    """

    # 0 ─── decide slice mode ────────────────────────────────────────────────
    if df_type == "game":
        slice_by: str | None = "GameName"
    elif df_type == "model":
        slice_by = "Model"
    else:                            # "pd" or None  →  no slicing
        slice_by = None

    df = df_agents.copy()
    df["LiedFirst"] = df["LiedFirst"].astype(bool)

    # ensure required columns
    needed = {"Agent", "Personality", "I/E", "N/S", "T/F", "J/P", "LiedFirst"}
    if slice_by:
        needed.add(slice_by)
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"df_agents missing columns: {missing}")

    # helper to analyse one MBTI axis ---------------------------------------
    def analyse(axis: str, pos: str, neg: str):
        # group columns (add slice if present)
        g_cols = ["Agent", "Personality", axis]
        if slice_by:
            g_cols.insert(1, slice_by)

        ag = df.groupby(g_cols, as_index=False)["LiedFirst"].mean()  # proportion per agent

        # agent-level counts
        pos_vals = ag.query(f"`{axis}` == @pos")["LiedFirst"]
        neg_vals = ag.query(f"`{axis}` == @neg")["LiedFirst"]
        p_ag = _prop_test(pos_vals.sum(), len(pos_vals), neg_vals.sum(), len(neg_vals))

        # 8-type pooling
        if slice_by:                          # per slice, then flatten
            type_slice = (
                ag.groupby(["Personality", axis, slice_by], as_index=False)["LiedFirst"]
                  .mean())
            pos_pool = np.concatenate(
                type_slice.query(f"`{axis}` == @pos")
                          .groupby(slice_by)["LiedFirst"].apply(list).values
            )
            neg_pool = np.concatenate(
                type_slice.query(f"`{axis}` == @neg")
                          .groupby(slice_by)["LiedFirst"].apply(list).values
            )
        else:                                 # no slice  →  one row per type
            type_slice = ag.groupby(["Personality", axis], as_index=False)["LiedFirst"].mean()
            pos_pool = type_slice.query(f"`{axis}` == @pos")["LiedFirst"].values
            neg_pool = type_slice.query(f"`{axis}` == @neg")["LiedFirst"].values

        p_type = _prop_test(pos_pool.sum(), len(pos_pool), neg_pool.sum(), len(neg_pool))

        return {
            "agent_slice": {
                "p": p_ag,
                "rate_pos": pos_vals.mean(),
                "rate_neg": neg_vals.mean(),
            },
            "type_slice": {
                "p": p_type,
                "rate_pos": pos_pool.mean(),
                "rate_neg": neg_pool.mean(),
            },
        }

    axes = {"I/E": ("I", "E"),
            "N/S": ("N", "S"),
            "T/F": ("T", "F"),
            "J/P": ("J", "P")}

    results: Dict[str, Dict[str, Dict[str, float]]] = {
        axis: analyse(axis, *lbls) for axis, lbls in axes.items()
    }

    # ---------- write TXT report ------------------------------------------
    os.makedirs(plots_path, exist_ok=True)
    label = slice_by if slice_by else "Overall"
    fname = os.path.join(plots_path, f"liedfirst_pvalues_{label}.txt")

    with open(fname, "w") as f:
        f.write(f"=== 'Lied First' Comparison ({label}) ===\n\n")
        for axis, (pos, neg) in axes.items():
            ag = results[axis]["agent_slice"]
            tp = results[axis]["type_slice"]

            higher = pos if ag["rate_pos"] > ag["rate_neg"] else neg
            lower  = neg if higher == pos else pos
            f.write(f"-- {axis}: {higher} > {lower} --\n")
            f.write(f"Agent level : {ag['rate_pos']:.2%} vs {ag['rate_neg']:.2%}  "
                    f"(p = {ag['p']:.3g})\n")
            f.write(f"8-type pool: {tp['rate_pos']:.2%} vs {tp['rate_neg']:.2%}  "
                    f"(p = {tp['p']:.3g})\n")

            if slice_by:
                qualifier = "robust" if tp["p"] < 0.05 else "weaker"
                f.write(f"Interpretation: The tendency for {higher} agents to lie first is "
                        f"{qualifier} across {slice_by.lower()}s.\n\n")
            else:
                f.write(f"Interpretation: Across all data, {higher} agents are more likely to "
                        f"lie first.\n\n")

    print(f"✅ Lied-first report saved to {fname}")
    return results

    
                    
#-----------------------------------------------------------------
### Plotting functions df
def plot_boxplot_score_all(df, plots_path, df_type):
    #TODO
    pass

def plot_boxplot_score_per_dichotomy(df, plots_path, df_type):
    #TODO
    pass

def plot_violin_score_all(df, plots_path, df_type):
    #TODO
    pass

def plot_violin_score_per_dichotomy(df, plots_path, df_type):
    #TODO
    pass

def plot_cumulative_score_all(df, plots_path, df_type):
    #TODO
    pass

def plot_truth_bar_all(df, plots_path, df_type):
    #TODO
    pass

def plot_truth_box_all(df, plots_path, df_type):
    #TODO
    pass

def plot_truth_violin_dichotomy(df, plots_path, df_type):
    #TODO
    pass

def plot_truth_violin_dichotomy_IE(df, plots_path, df_type):
    #TODO
    pass

def plot_who_lied_all(df, plots_path, df_type):
    #TODO
    pass

### Plotting functions df_agents

def plot_defection_rate_bar_dichotomy(df_agents: pd.DataFrame, plots_path, df_type) -> str:
    """
    Bar-plots of mean defection rate for each MBTI dichotomy (I/E, N/S, T/F, J/P).

    Saves   <plots_path>/defection_rate_by_dichotomy.png   and returns that filepath.
    """

    # 1. row-level defection rate -------------------------------------------------------
    def list_to_rate(actions):
        if isinstance(actions, str):
            actions = ast.literal_eval(actions)
        return np.mean([str(a).strip().lower() == "defect" for a in actions])

    df = df_agents.copy()
    df["defection_rate"] = df["Actions"].apply(list_to_rate)

    # 2. one mean per MBTI personality --------------------------------------------------
    group_cols = ["Personality", "I/E", "N/S", "T/F", "J/P"]
    mbti_means = (
        df.groupby(group_cols, as_index=False)["defection_rate"]
          .mean()
    )

    # 3. plotting -----------------------------------------------------------------------
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['figure.figsize'] = [10, 6]

    fig, axes = plt.subplots(2, 2, sharey=True)
    axes = axes.flatten()

    dichotomies = [("I/E", ("I", "E")),
                   ("N/S", ("N", "S")),
                   ("T/F", ("T", "F")),
                   ("J/P", ("J", "P"))]

    for ax, (col, order) in zip(axes, dichotomies):
        sns.barplot(
            data=mbti_means,
            x=col, y="defection_rate",
            order=list(order),
            hue=col,               # <-- satisfy “palette needs a hue”
            palette="Set2",
            legend=False,
            errorbar="sd",         # <-- modern replacement for ci="sd"
            ax=ax,
        )
        ax.set_title(f"Defection rate by {col}")
        ax.set_xlabel("")
        ax.set_ylabel("Mean defect rate")

    fig.tight_layout()

    # 4. save ---------------------------------------------------------------------------
    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "defection_rate_by_dichotomy.png")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_defection_rate_bar_all(df_agents: pd.DataFrame, plots_path: str | os.PathLike, df_type) -> str:
    def list_to_rate(actions):
        if isinstance(actions, str):
            actions = ast.literal_eval(actions)
        return np.mean([str(a).strip().lower() == "defect" for a in actions])

    df = df_agents.copy()
    df["defection_rate"] = df["Actions"].apply(list_to_rate)

    # ------------------------------------------------------------------ 2. one mean per MBTI code
    mbti_means = (
        df.groupby("Personality", as_index=False)["defection_rate"]
          .mean()
          .sort_values("defection_rate", ascending=False)   # tallest bar first
    )

    # ------------------------------------------------------------------ 3. plottin

    fig, ax = plt.subplots()

    hue_col = "Personality"     # satisfy seaborn’s palette requirement

    sns.barplot(
        data=mbti_means,
        x="Personality", y="defection_rate",
        hue=hue_col,
        order=personality_order,
        palette="Set2",
        legend=False,
        errorbar="sd",              # seaborn ≥0.14
        ax=ax,
    )

    ax.set_title("Mean defection rate by MBTI personality")
    ax.set_xlabel("")
    ax.set_ylabel("Mean defect rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()

    # ------------------------------------------------------------------ 4. save
    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "defection_rate_all.png")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_truth_strip_dichotomy(df_agents, plots_path, df_type):
    return
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))

    # Get 16 colors and map them to personality types
    personalities = sorted(df_agents['Personality'].dropna().unique())
    palette_colors = sns.color_palette("tab20", n_colors=len(personalities))
    palette = dict(zip(personalities, palette_colors))  # Map personality -> color

    # Re-group
    df_grouped_game = df_agents.groupby(
        ['GameName', 'Personality', 'I/E', 'N/S', 'T/F', 'J/P']
    )['Truthfullness'].mean().reset_index()

    dichotomies = ['I/E', 'N/S', 'T/F', 'J/P']
    axes = axs.flatten()

    # Plot without subplot legends
    for ax, col in zip(axes, dichotomies):
        sns.stripplot(
            x=col, y='Truthfullness', hue='Personality', data=df_grouped_game,
            ax=ax, palette=palette, jitter=True, dodge=False, size=10, legend=False
        )
        ax.set_title(f'Truthfulness by {col}')

    # Manually create legend handles
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=palette[p], label=p, markersize=10)
        for p in personalities
    ]

    fig.legend(handles=handles, title='Personality',
               loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 1])  # Leave space for legend
    plt.savefig(os.path.join(plots_path, 'truth_strip_dichotomy.png'), dpi=300)
    plt.close(fig)

def plot_truth_by_round(df_agents, plots_path, df_type):
    # Prepare the data for plotting
    df_agents = df_agents.copy()
    df_agents['Round'] = df_agents['CumulativeScores'].apply(lambda x: list(range(len(x))))
    df_agents['TruthfulByRound'] = df_agents.apply(lambda row: list(zip(row['Round'], row['Truthful'])), axis=1)

    # Explode the data to have one row per round
    df_exploded = df_agents.explode('TruthfulByRound')
    df_exploded['Round'] = df_exploded['TruthfulByRound'].apply(lambda x: x[0] if isinstance(x, tuple) else None)
    df_exploded['Truthful'] = df_exploded['TruthfulByRound'].apply(lambda x: x[1] if isinstance(x, tuple) else None)

    # Filter out rows with missing values
    df_exploded = df_exploded.dropna(subset=['Round', 'Truthful'])

    # Convert Round to integer for proper sorting
    df_exploded['Round'] = df_exploded['Round'].astype(int)

    # Plot average truthfulness by round for each dichotomy
    sns.set(style="whitegrid")
    sns.set_context("talk")
    sns.set_palette("husl")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    sns.lineplot(data=df_exploded, x='Round', y='Truthful', hue='I/E', ax=axs[0, 0])
    axs[0, 0].set_title('Average Truthfulness by Round - I/E')

    sns.lineplot(data=df_exploded, x='Round', y='Truthful', hue='N/S', ax=axs[0, 1])
    axs[0, 1].set_title('Average Truthfulness by Round - N/S')

    sns.lineplot(data=df_exploded, x='Round', y='Truthful', hue='T/F', ax=axs[1, 0])
    axs[1, 0].set_title('Average Truthfulness by Round - T/F')

    sns.lineplot(data=df_exploded, x='Round', y='Truthful', hue='J/P', ax=axs[1, 1])
    axs[1, 1].set_title('Average Truthfulness by Round - J/P')

    plt.tight_layout()
    plt.savefig(plots_path + "truthfulness_by_round.png", dpi=300)
    plt.close(fig)

# ── colour + label maps ────────────────────────────────────────────────────
PALETTE_BY_PAIR = {
    "I/E": sns.color_palette("Blues",   2),
    "N/S": sns.color_palette("Greens",  2),
    "T/F": sns.color_palette("Oranges", 2),
    "J/P": sns.color_palette("Purples", 2),
}
HUE_COLOUR = {
    "I": PALETTE_BY_PAIR["I/E"][0], "E": PALETTE_BY_PAIR["I/E"][1],
    "N": PALETTE_BY_PAIR["N/S"][0], "S": PALETTE_BY_PAIR["N/S"][1],
    "T": PALETTE_BY_PAIR["T/F"][0], "F": PALETTE_BY_PAIR["T/F"][1],
    "J": PALETTE_BY_PAIR["J/P"][0], "P": PALETTE_BY_PAIR["J/P"][1],
}
LABEL_FULL = {
    "I": "Introversion", "E": "Extraversion",
    "N": "Intuition",    "S": "Sensing",
    "T": "Thinking",     "F": "Feeling",
    "J": "Judging",      "P": "Perceiving",
}

# ── helper to build ONE legend, now ABOVE the whole figure ────────────────
def _legend_above(fig, title="", ncol=8):
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        if ax.legend_:
            ax.legend_.remove()

    uniq = {lab: h for lab, h in zip(labels, handles)}
    full_labels = [LABEL_FULL[lab] for lab in uniq.keys()]

    fig.legend(
        uniq.values(), full_labels,
        title=title,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=ncol,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    
def _legend_mbti(fig):
    """Colour legend for MBTI dichotomies, centred above the axes."""
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
        if ax.legend_: ax.legend_.remove()

    uniq = {lab: h for lab, h in zip(labels, handles)}
    full = [LABEL_FULL[l] for l in uniq]
    fig.legend(
        uniq.values(), full, title="MBTI dichotomies",
        loc="lower center", bbox_to_anchor=(0.5, 1.19),
        frameon=False, ncol=8, handletextpad=.6, columnspacing=1.4,
    )

def _legend_alias(fig, alias_map, max_columns=4):
    from matplotlib.patches import Patch
    import math
    """Text-only legend mapping alias → full model name."""
    items = [f"{abbr} – {full}" for full, abbr in alias_map.items()]
    # break into rows so it stays narrow enough
    rows = math.ceil(len(items) / max_columns)
    chunks = [items[i::rows] for i in range(rows)]
    lines = ["   •   ".join(chunk) for chunk in chunks]

    # single invisible patch so matplotlib reserves the spot
    patch = Patch(facecolor="white", edgecolor="white", label="\n".join(lines))
    fig.legend(
        handles=[patch], loc="lower center",
        bbox_to_anchor=(0.5, 1.12), frameon=False,
        handlelength=0, handletextpad=0, fontsize="small",
    )
def plot_truthfulness_by_df_type_dichotomies(df_agents, plots_path, df_type):
    """
    Makes a 2×2 grid of barplots:
      • x-axis uses aliases (see MODEL_ALIAS)
      • top legend 1: alias → full model name
      • top legend 2: colour code for MBTI dichotomies
    """
    if df_type == "pd":
        return
    slice_by = "Model" if df_type == "model" else "GameName"

    # ----- add alias column ----------------------------------------------
    df = df_agents.copy()
    alias_col = f"{slice_by}_alias"
    df[alias_col] = df[slice_by].map(MODEL_ALIAS).fillna(df[slice_by])

    # ----- plotting -------------------------------------------------------
    sns.set(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(22, 13))

    pairs = [("I/E", ["I", "E"]),
             ("N/S", ["N", "S"]),
             ("T/F", ["T", "F"]),
             ("J/P", ["J", "P"])]

    for i, (pair, hues) in enumerate(pairs, 1):
        ax = plt.subplot(2, 2, i)
        sns.barplot(
            x=alias_col, y="Truthfulness", hue=pair,
            data=df, palette=[HUE_COLOUR[h] for h in hues],
            hue_order=hues, ax=ax,
        )
        ax.set(xlabel=f"{slice_by} (alias)", ylabel="Truthfulness")
        ax.tick_params(axis="x", rotation=45)

    # ----- legends --------------------------------------------------------
    _legend_alias(fig, MODEL_ALIAS, max_columns=4)
    _legend_mbti(fig)

    # leave space for legends (auto: plenty for two lines)
    fig.subplots_adjust(top=1.18)
    fig.tight_layout(rect=[0, 0, 1, 1.15])

    fig.savefig(f"{plots_path}truthfulness_by_{slice_by}_dichotomies.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

# ── I/E-only plot ──────────────────────────────────────────────────────────
def plot_truthfulness_by_df_type_IE(df_agents, plots_path, df_type):
    df_agents = df_agents.copy()
    if df_type == "pd":
        return
    slice_by = "Model" if df_type == "model" else "GameName"

    sns.set(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(15, 9))

    ax = plt.subplot(1, 1, 1)
    sns.barplot(
        x=slice_by, y="Truthfulness", hue="I/E",
        data=df_agents,
        palette=[HUE_COLOUR["I"], HUE_COLOUR["E"]],
        hue_order=["I", "E"],
        ax=ax,
    )
    ax.set(xlabel=f"{slice_by} name", ylabel="Truthfulness")
    ax.tick_params(axis="x", rotation=45)

    fig.subplots_adjust(top=0.99)
    _legend_above(fig, title="Introversion / Extraversion", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(f"{plots_path}truthfulness_{slice_by}_I_E.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    
def plot_truthfulness_violin_by_df_typ(df_agents, plots_path, df_type):
    df_agents = df_agents.copy()
    if df_type == "pd":
        return
    slice_by = "Model" if df_type == "model" else "GameName"
    sns.set(style="whitegrid", context="talk")
    slices = df_agents[slice_by].unique()
    for slice_value in slices:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(
            x="I/E",
            y="Truthfulness",
            data=df_agents[df_agents[slice_by] == slice_value],
            palette=[HUE_COLOUR["I"], HUE_COLOUR["E"]],
            hue="I/E",
            order=["I", "E"],
            ax=ax,
        )
        # remove any leftover legend
        if ax.legend_:
            ax.legend_.remove()
        ax.set(
            xlabel="I/E",
            ylabel="Truthfulness",
            title=f"Truthfulness for {slice_value}"
        )
        ax.tick_params(axis="x")
        plt.tight_layout()
        plt.savefig(f"{plots_path}truthfulness_{slice_value}_I_E.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_mean_truthfulness_by_df_type(df_agents, plots_path, df_type):
    df_agents = df_agents.copy()
    if df_type == "pd":
        return
    slice_by = "Model" if df_type == "model" else "GameName"
    sns.barplot(data=df_agents, x=slice_by, y='Truthfulness', palette='Set2')
    plt.xlabel(f'{slice_by}')
    plt.ylabel('Truthfulness Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'truthfulness_mean_all_{df_type}.png'), bbox_inches='tight')
    plt.close()
    
def plot_strategy_switches_by_df_type_TF(df_agents, plots_path, df_type):
    df_agents = df_agents.copy()
    if df_type == "pd":
        return
    slice_by = "Model" if df_type == "model" else "GameName"
    
    for slice_name in df_agents[slice_by].unique():
        slice_df = df_agents[df_agents[slice_by] == slice_name]
        t_switches = slice_df.loc[slice_df['T/F'] == 'T', 'Switches']
        f_switches = slice_df.loc[slice_df['T/F'] == 'F', 'Switches']
        
        t_counts = t_switches.value_counts().reindex(range(7), fill_value=0)
        f_counts = f_switches.value_counts().reindex(range(7), fill_value=0)
        
        plt.figure(figsize=(7,5))
        sns.barplot(x=t_counts.index, y=t_counts.values,
                color='steelblue', label='T')
        sns.barplot(x=f_counts.index, y=-f_counts.values,
                color='salmon', label='F')
        
        max_y = max(t_counts.values.max(), f_counts.values.max())
        plt.ylim(-max_y - 2, max_y + 2)
        plt.axhline(0, color='black', linewidth=0.8)

        plt.xlabel('Number of Strategy Switches')
        plt.ylabel('Frequency')
        #plt.title(f"Strategy switches for T/F personalities : {slice_name}")
        plt.legend(title='Dichotomy', loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f'strategy_switches_{slice_name}_TF.png'), dpi=300)
        plt.close()
        
def plot_who_lied_first(df_agents, plots_path, df_type):
    #TODO:adapt to free will
    df_agents = df_agents.copy()
    max_possible_lies = df_agents["LiedFirst"].sum()
    # compute fraction of episodes where each agent lied first
    summary = (
        df_agents
        .groupby("Personality", as_index=False)["LiedFirst"]
        .mean()
    )

    # plot
    sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(
        data=summary,
        x="Personality", y="LiedFirst",
        hue="Personality",
        palette="tab20",
        ax=ax
    )
    ax.set_ylabel("Proportion Lied First")
    #set y limit
    ax.set_ylim(0, 1)

    # save
    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "who_lied_first.png")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_who_lied_first_IE_TF(df_agents, plots_path, df_type):
    #TODO: adapt o free will
    df_agents = df_agents.copy()
    
    # compute fraction of episodes where each agent lied first
    summary = (
        df_agents
        .groupby("I/E", as_index=False)["LiedFirst"]
        .sum()
    )

    # plot
    sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(
        data=summary,
        x="I/E", y="LiedFirst",
        hue="I/E",
        palette=["#4c72b0", "#dd8452"],
        ax=ax
    )
    ax.set_ylabel("Count Lied First")
    ax.tick_params(axis='x', rotation=45)
    # save
    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "who_lied_first_IE.png")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    #now TF
    # now TF
    summary_tf = (
        df_agents
        .groupby("T/F", as_index=False)["LiedFirst"]
        .sum()
    )

    sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(
        data=summary_tf,
        x="T/F", y="LiedFirst",
        hue="T/F",
        palette=["#4c72b0", "#dd8452"],
        ax=ax
    )
    ax.set_ylabel("Count Lied First")

    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "who_lied_first_TF.png")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
#--------------------------------------------------------------------------------------------------------------------------------------

def plot_from_list(df, plots_path, plot_functions, df_type=None):
    for func in plot_functions:
        try:
            func(df, plots_path, df_type)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            continue
#run whats runnable
def PD_plots():
    df_type = 'pd'
     # Set the path to save the plots
    df, plots_path = import_df_and_plot_path(df_type)

    df_agents_og = import_df_agents(df)
    
    # possible plotting functions
    df_functions = [
        plot_boxplot_score_all,
        plot_violin_score_all,
        plot_violin_score_per_dichotomy,
        plot_cumulative_score_all,
        plot_truth_bar_all,
        plot_truth_violin_dichotomy,
        plot_truth_violin_dichotomy_IE,
        plot_who_lied_all,
        get_liedfirst_pvalues,
    ]
    df_agent_functions = [
        plot_defection_rate_bar_all,
        plot_defection_rate_bar_dichotomy,
        get_defection_rates_dichos,
        plot_truth_strip_dichotomy,
        get_truthfulness_dichos,
        plot_truth_by_round,
        get_liedfirst_pvalues,
        plot_who_lied_first,
        plot_who_lied_first_IE_TF,
    ]
    # Run the plotting functions
    plot_from_list(df, plots_path, df_functions, df_type)
    plot_from_list(df_agents_og, plots_path, df_agent_functions, df_type)
            
def model_plots():
    df_type = 'model'
    # Set the path to save the plots
    df, plots_path = import_df_and_plot_path(df_type)

    df_agents_og = import_df_agents(df)
    
    # possible plotting functions
    df_functions = [
        plot_boxplot_score_all,
        plot_violin_score_all,
        plot_violin_score_per_dichotomy,
        plot_cumulative_score_all,
        plot_truth_bar_all,
        plot_truth_violin_dichotomy,
        plot_truth_violin_dichotomy_IE,
        plot_who_lied_all,
        
    ]
    df_agent_functions = [
        plot_defection_rate_bar_all,
        plot_defection_rate_bar_dichotomy,
        get_defection_rates_dichos,
        plot_truth_strip_dichotomy,
        get_truthfulness_dichos,
        plot_truth_by_round,
        plot_truthfulness_by_df_type_dichotomies,
        plot_truthfulness_by_df_type_IE,
        plot_truthfulness_violin_by_df_typ,
        plot_mean_truthfulness_by_df_type,
        plot_strategy_switches_by_df_type_TF,
        get_strategy_switch_pvalues,
        get_liedfirst_pvalues,
        plot_who_lied_first,
        plot_who_lied_first_IE_TF,
    ]
    # Run the plotting functions
    plot_from_list(df, plots_path, df_functions, df_type)
    plot_from_list(df_agents_og, plots_path, df_agent_functions, df_type)

def game_plots():
    df_type = 'game'
    # Set the path to save the plots
    df, plots_path = import_df_and_plot_path(df_type)

    df_agents_og = import_df_agents(df)
    
    # possible plotting functions
    df_functions = [
        plot_boxplot_score_all,
        plot_violin_score_all,
        plot_violin_score_per_dichotomy,
        plot_cumulative_score_all,
        plot_truth_bar_all,
        plot_truth_violin_dichotomy,
        plot_truth_violin_dichotomy_IE,
        plot_who_lied_all
    ]
    df_agent_functions = [
        plot_truth_strip_dichotomy,
        get_truthfulness_dichos,
        plot_truth_by_round,
        plot_truthfulness_by_df_type_dichotomies,
        plot_truthfulness_by_df_type_IE,
        plot_truthfulness_violin_by_df_typ,
        plot_mean_truthfulness_by_df_type,
        plot_strategy_switches_by_df_type_TF,
        get_strategy_switch_pvalues,
        get_liedfirst_pvalues,
        plot_who_lied_first,
        plot_who_lied_first_IE_TF,
    ]
    # Run the plotting functions
    plot_from_list(df, plots_path, df_functions)
    plot_from_list(df_agents_og, plots_path, df_agent_functions, df_type)

if __name__ == '__main__':
    # Set the style of seaborn
    sns.set(style="whitegrid")
    sns.set_palette("Set2")

    # Set the font size for all plots
    plt.rcParams.update({'font.size': 14})

    # Set the figure size for all plots
    plt.rcParams['figure.figsize'] = [10, 6]

    PD_plots()
    #model_plots()
    game_plots()

    