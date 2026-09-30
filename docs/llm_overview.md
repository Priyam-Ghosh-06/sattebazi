# Team overviews with a local open-weight LLM (optional bonus)

`src/shared/llm/team_overview.py` turns each predicted XI of the output CSV into a short paragraph.

```bash
python src/shared/llm/team_overview.py --predictions results/hidden_predictions.csv --output results/hidden_team_overviews.md
```

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` by default (Apache-2.0, open weights, ~1 GB, runs on CPU in a few seconds per fixture);
  any open-weight chat model on the Hugging Face hub can be passed with `--model`. No commercial API is called.
- **Facts first, words second.** The code computes the facts (composition in full role names, the team split, the three
  highest expected scorers, total expected points, players with P(play) < 0.75) and the LLM only phrases them. The facts are
  printed verbatim above every paragraph, so a wording slip by a small model is visible at once.
- **Reproducible:** greedy decoding and a fixed seed; the same CSV gives the same text.

Example (mock fixtures): [`results/mock_team_overviews.md`](../results/mock_team_overviews.md).
