"""Optional bonus (PS): a short text overview of each predicted XI, written by a local open-weight LLM.

python src/shared/llm/team_overview.py --predictions results/hidden_predictions.csv --output results/hidden_team_overviews.txt

The model only phrases facts computed here (role counts, team split, top picks, playing-probability risks); the facts are also
printed verbatim above each overview, so any wording error is visible. Default model: Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0,
open weights, runs on CPU). Greedy decoding and a fixed seed make the text reproducible. No API calls.
"""
import argparse, os, random

import numpy as np
import pandas as pd

random.seed(0); np.random.seed(0)


def facts(g):
    """Deterministic facts about one predicted XI (rows of the PS CSV)."""
    names = {"WK": ("wicket-keeper", "wicket-keepers"), "BAT": ("batter", "batters"), "AR": ("all-rounder", "all-rounders"),
             "BOWL": ("bowler", "bowlers")}
    roles = g.role.value_counts().reindex(list(names)).fillna(0).astype(int)
    split = g.player_team.value_counts()
    top = g.nlargest(3, "predicted_points")
    f = {"fixture": f"{g.team1.iloc[0]} vs {g.team2.iloc[0]} on {g.match_date.iloc[0]}",
         "composition": ", ".join(f"{n} {names[r][n != 1]}" for r, n in roles.items() if n),
         "team_split": ", ".join(f"{n} from {t}" for t, n in split.items()),
         "top_picks": "; ".join(f"{r.player} ({r.player_team}, {names[r.role][0]}, {r.predicted_points:.1f} expected points)"
                                for r in top.itertuples()),
         "total_predicted": f"{g.predicted_points.sum():.1f}"}
    if "p_plays" in g and g.p_plays.notna().any():
        risky = g[g.p_plays < 0.75]
        f["selection_risks"] = "; ".join(f"{r.player} (plays with p = {r.p_plays:.2f})" for r in risky.itertuples()) or "none below 0.75"
    return f


def prompt(f):
    lines = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in f.items())
    return ("You are a cricket fantasy analyst. Using ONLY the facts below, write a 3-4 sentence overview of this predicted "
            "fantasy XI: explain its composition, highlight the key players and mention any selection risk. Do not invent "
            f"statistics.\n\nFacts:\n{lines}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="results/hidden_predictions.csv")
    ap.add_argument("--output", default="results/hidden_team_overviews.txt")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="any open-weight chat model on the Hugging Face hub")
    ap.add_argument("--max-fixtures", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    lm.eval()
    P = pd.read_csv(a.predictions)
    groups = list(P.groupby(["match_date", "team1", "team2"], sort=False))[: a.max_fixtures]
    out = [f"Predicted XI overviews, written by {a.model} (open weights, local, greedy decoding) from {a.predictions}\n"]
    for _, g in groups:
        f = facts(g)
        msgs = [{"role": "user", "content": prompt(f)}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False)
        text = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        out.append(f"=== {f['fixture']} ===\n" + "\n".join(f"  {k.replace('_', ' ')}: {v}" for k, v in f.items() if k != "fixture")
                   + f"\n\n{text}\n")
        print(f"[llm] {f['fixture']}", flush=True)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    open(a.output, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print(f"wrote {a.output}: {len(groups)} overviews")


if __name__ == "__main__":
    main()
