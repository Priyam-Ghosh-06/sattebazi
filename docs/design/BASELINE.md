# Caveman baseline (CB)

Status: design, **revision 3** (2026-09-29, night). Supersedes revision 2 (the "neural baseline" NB with a CT-WKV encoder and capacity binding). Those two components are not deleted: they move to `ADVANCED.md` §A0 as the first rungs of the advanced ladder. **Do not implement until §12 is closed.**
Place in the pipeline: model stage, rung CB (`PIPELINE.md` §4). It is compared with the staleness-aware B0 (E16) and is the reference every ADV rung is measured against.

---

## 0. What CB is

One learned map from raw, pre-toss information to a number per candidate:

$$
f_\theta:\ \big(\text{fixture},\ \{\text{raw history of } i\}_{i\in S_A\cup S_B}\big)\ \longmapsto\ \hat Y \in \mathbb R^{|S_A|+|S_B|},
$$

trained with squared error on the realised fantasy score $Y_i$ (0 for a squad member who did not play). The pipeline is: **preprocessing → history encoder → squad decoder → $\hat Y$ → rule-mandated selection → CSV**.

The model is asked to learn every mapping internally: who is a keeper or a bowler, who is a regular, how form decays, how stale the information is, how a venue or opponent shifts scoring, and how many of the squad actually play. It is given no hand-built structure.

| not in CB | where it comes back |
|---|---|
| split into $p$ (plays) and $\mu$ (points if plays) | ADV A0b |
| capacity vectors (keeper, bowling option, batting depth, overseas) | ADV A0b, A3 |
| a decay law in time (CT-WKV) | ADV A0a |
| sum-to-XI shift τ, constrained-choice layer | ADV A0b, A3 |
| Φ components as separate targets, innings simulator | ADV A4 |
| roll-forward across the staleness gap | ADV A5 |

What CB keeps is not modelling structure. Each item is either data correctness or a rule of the problem statement:

| kept | why it is not "structure" |
|---|---|
| history truncated at the cutoff $c$, staleness $\Delta$ as an input, Δ sampled in training | without it the model trains on fresh history and is tested on stale history: a distribution-shift bug, not a modelling choice |
| season pseudo-squads as candidate sets | without them there are no zero targets at all (benched players never appear in Cricsheet) |
| fantasy roles (WK/BAT/AR/BOWL) inferred from history | the PS requires a role per output player and composition bounds; roles are used **only** by the ILP, never as model input |
| exact ILP | the PS requires a valid XI; the ILP is the exact argmax of the model's own objective under those rules |

---

## 1. Target, and why squared error is enough

For candidate $i$ in fixture $m$ let $D_i = \mathbb 1[i\in X_m]$ and

$$
Y_i = D_i\,\big(3 + \Phi(E_i)\big).
$$

Let $\mathcal I$ be everything the model sees (both squads' histories truncated at $c$, the fixture, Δ).

**Proposition CB-1 (the product is learned without being named).**
$\arg\min_f E\big[(Y_i - f(\mathcal I))^2\big] = E[Y_i\mid\mathcal I] = p_i\,\mu_i$ with $p_i = P(D_i = 1\mid\mathcal I)$ and $\mu_i = E[3+\Phi\mid D_i = 1,\mathcal I]$.

*Proof.* For any $f$, $E[(Y-f)^2\mid\mathcal I] = (f - E[Y\mid\mathcal I])^2 + \operatorname{Var}(Y\mid\mathcal I)$, since the cross term has conditional mean 0. The second term does not depend on $f$, so the minimiser is $E[Y\mid\mathcal I]$. By the tower property over $D$: $E[Y\mid\mathcal I] = P(D=1\mid\mathcal I)\,E[Y\mid D=1,\mathcal I] + P(D=0\mid\mathcal I)\cdot 0 = p\,\mu$. ∎

So CB and a factorised model have the **same population optimum**. The difference between them is purely statistical: inductive bias and sample efficiency. That is exactly what the ADV ladder measures.

**Why this is also the right thing to select on.** The fantasy objective is $E[\sum_i x_iY_i\mid\mathcal I] = \sum_i x_i E[Y_i\mid\mathcal I]$ by linearity, with no independence assumption. CB outputs $E[Y_i\mid\mathcal I]$ directly, so the ILP needs nothing else.

**How hard the target is (variance decomposition).** By the law of total variance over $D$,

$$
\operatorname{Var}(Y\mid\mathcal I) = \underbrace{p\,\sigma_\mu^2}_{\text{performance noise}} + \underbrace{p(1-p)\,\mu^2}_{\text{selection noise}},\qquad \sigma^2_\mu = \operatorname{Var}(3+\Phi\mid D=1,\mathcal I).
$$

With ICC(Φ) ≈ 0.13 (E2), $\sigma_\mu$ is large; for a rotation player ($p\approx 0.5$, $\mu\approx 30$) the selection term alone is $0.25\cdot 900 = 225$ points². Most of the squared error is irreducible, which is why the pre-registered expectation (§8) is a tie with B0.

**MAE.** The MAE-optimal prediction is the conditional median, which is 0 whenever $p < 0.5$. CB outputs the mean, which is right for selection. The PS's MAE is reported for the mean (and, for comparison, for a separately trained L1 head, `--loss l1`).

---

## 2. Preprocessing: data to tensors

### 2.1 Rows and targets
A training example is one fixture $m$ with both teams' candidate sets: the **season pseudo-squads** $\hat S_A, \hat S_B$ from `src/squads.py` (everyone who played for the team in that event and season). For each candidate: target $y_i = D_i(3+\Phi_i)$, stored in `player_match.parquet` as `total` for players and 0 for benched candidates. At test time the candidates are the given squads.

Real example (IPL 2025, LSG v CSK, 14 Apr, match 1473467): CSK pseudo-squad 22 players, 12 played (impact sub included); LSG 20, 12 played.

### 2.2 Cutoff and staleness
Each training fixture draws $\Delta\sim\mathcal D_\Delta$ (PIPELINE D3: mass 0.10 at 0, 0.40 on U[1, 270], 0.50 on U[271, 450] days), fresh each epoch, seed = hash(fixture, epoch). Cutoff $c = \text{date}(m) - \Delta$. **Every token must be dated strictly before $c$** (test T1). At test time $c$ = 2025-12-31.

### 2.3 History tokens (per candidate)
Two kinds of token, each a raw fact:

- **appearance**: one per match $i$ played (any team, T20/IT20), dated $< c$.
- **absence**: one per match of the *fixture team* $T$ in which $i$ did not play, dated in $[\max(t^{\text{first}}_{iT},\ c - 365\text{ d}),\ c)$, where $t^{\text{first}}_{iT}$ is $i$'s first appearance for $T$. No membership is inferred beyond "has appeared for $T$ before". This is what lets the model see that a player is a regular, a rotation option or dropped (see the CSK 2025 appearance strips in the dashboard).

Keep the $L = 64$ most recent tokens (sorted by date, then match ID; stable).

| field group | content | transform |
|---|---|---|
| kind | appearance / absence | embedding |
| raw counts (16) | runs, balls faced, 4s, 6s, dismissed, bat position (0 = DNB), legal balls bowled, runs conceded, dots, credited wickets, lbw/bowled, maidens, catches, stumpings, run-outs direct, indirect | `log1p`, standardised; 0 on absence tokens |
| realised points | $3+\Phi$ of that match | standardised; 0 on absence tokens |
| context | format, gender, team_type, event, venue, opponent, team played for, `is_fixture_team`, batted first, won, scheduled overs, balls per over | embeddings (vocabulary: frequency ≥ 20 in the training fold, else UNK); scalars standardised |
| time | age $a_j = \log(1 + (c - t_j)/\text{1 day})$; gap to previous token $\log(1+\delta_j)$ | Fourier features of $a_j$ (8 frequencies) + raw scalars |

Standardisation statistics and vocabularies come from the training fold only.

### 2.4 Fixture token
Format, gender, team_type, event, venue, month, scheduled overs, balls per over, $\log(1+\Delta)$.

### 2.5 Cold start
A candidate with no tokens gets only the summary token (§3) plus the scalar $\log(1+n_{\text{tokens}}) = 0$; debutants appear in training as zero-history candidates, so this case is learned.

### 2.6 Roles (for the ILP only)
Fantasy role of each candidate as of $c$ from `src/roles.py` (unchanged). UNK counts as BAT in the ILP (DECISIONS.md).

---

## 3. History encoder $E_\phi$ (one per candidate, shared weights)

Token embedding, $d = 64$:

$$
x_j = W_{\text{num}}\,\text{num}_j + \sum_{k}\text{Emb}_k(\text{cat}_{jk}) + W_t\,\gamma(a_j) + \text{Emb}_{\text{rank}}(\text{recency rank}_j),
$$

with $\gamma(a) = [\sin(2\pi\omega_ka), \cos(2\pi\omega_ka)]_{k=1}^{8}$, $\omega_k$ log-spaced.

Sequence $[\,\texttt{SUM}, x_1, \dots, x_{L_i}\,]$ → 2 layers of pre-LayerNorm Transformer encoder (4 heads, FFN $4d$ with GELU, dropout 0.1), key-padding mask for $L_i < 64$. Attention is bidirectional **within** the history: every token is before $c$, so no token sees the future of the fixture. Output: $h_i$ = final state at `SUM`, plus $\log(1+n_{\text{tokens}})$ concatenated and projected back to $d$.

For one head: $Q = XW_Q$, $K = XW_K$, $V = XW_V$, $\operatorname{Attn}(X) = \operatorname{softmax}\big(QK^\top/\sqrt{d_k} + M\big)V$, with $M_{jk} = -\infty$ on padding.

## 4. Squad decoder $D_\psi$ (one per fixture)

Set of tokens: $z_0 = g_m$ (fixture token, embedded as §2.4) and $z_i = h_i + e_{\text{side}(i)}$ for every candidate in both squads, with learned side embeddings $e_A, e_B$. **A/B swap augmentation**: in training, which team is "A" is drawn at random each time, so the side embeddings can only encode "same team as me / other team", not "team1".

2 layers of pre-LayerNorm Transformer over the set, **no positional encoding**, key-padding mask. Head: $\hat Y_i = \sigma_Y\, w^\top\operatorname{MLP}(z_i^{\text{out}}) + \bar y$ ($\bar y, \sigma_Y$ from the training fold).

It is called a decoder because it turns representations into the output; it is not autoregressive.

**Proposition CB-2 (permutation equivariance).** Relabelling the candidates permutes the outputs the same way and changes nothing else.
*Proof.* For a permutation matrix $\Pi$: $\operatorname{softmax}(\Pi QK^\top\Pi^\top)\Pi V = \Pi\operatorname{softmax}(QK^\top)\Pi^\top\Pi V = \Pi\operatorname{Attn}(Z)$, because row-wise softmax commutes with simultaneous row/column permutation and $\Pi^\top\Pi = I$. LayerNorm, FFN and the head act row-wise. ∎ (Test T4.)

**What the decoder can represent that a per-candidate model cannot.** $\hat Y_i$ may depend on who else is in the squad: with two keepers, the backup's output can rise when the first-choice keeper is missing. CB can *represent* this; nothing forces it to learn it, and nothing forces $\sum_i \hat p_i = 11$. ADV A3 makes both structural.

## 5. Loss

$$
\mathcal L(\theta) = \frac{1}{|\mathcal B|}\sum_{m\in\mathcal B}\frac{1}{|S_m|}\sum_{i\in S_m}\Big(\frac{\hat Y_i - y_i}{\sigma_Y}\Big)^2,\qquad \frac{\partial\mathcal L}{\partial\hat Y_i}\propto \hat Y_i - y_i .
$$

A benched candidate pulls $\hat Y_i$ toward 0; a player pulls it toward his realised $3+\Phi$. Averaged over many fixtures with the same inputs, the pull balances at $p\mu$ (Proposition CB-1). `--loss huber` (δ tuned) exists for parity with E5, where Huber tied.

## 6. Training

- **Folds**: walk-forward, test years 2024 and 2025 (full years). Training fixtures 2019-01-01 → Y−1; tokens from all history before each example's $c$. Validation: last 3 months of the training window at the test-matching $\mathcal D_\Delta$.
- **Batches**: 32 fixtures (both squads of a fixture always together; the decoder needs the whole set).
- **Optimiser**: AdamW, peak lr 1e-3, β = (0.9, 0.98), weight decay 0.05 on weight matrices only, 5% linear warm-up then cosine to 2% of peak, gradient-norm clip 1.0, dropout 0.1 (MLP/attention) and 0.05 (embeddings), EMA of weights 0.999 for validation and test.
- **Stopping**: ≤ 30 epochs, patience 5, on validation captured/oracle at the test-matching Δ, ties broken by validation loss.
- **Grid** (2024 fold only, then frozen): $d\in\{32, 64\}$, layers per block ∈ {1, 2}, lr ∈ {5e-4, 1e-3}, 3 seeds each; report seed spread.
- **Reproducibility**: sweeps on Kaggle; the final model is trained by `src/train.py --arch cb` on CPU with `torch.use_deterministic_algorithms(True)`, fixed threads, DataLoader workers = 0. About 2×10⁵ parameters.

## 7. Inference (`src/evaluate.py --arch cb`)

For each fixture: resolve names to IDs (existing resolver; rate printed) → tokens dated ≤ 2025-12-31 → encoder per candidate → decoder over both squads with $\Delta$ = fixture date − 2025-12-31 → $\hat Y$ → ILP with roles as of $c$ → CSV, predicted points $= \hat Y_i$.

## 8. Evaluation and decision rule (pre-registered)

Metrics as in `PIPELINE.md` §5 at Δ ∈ {0, 90, 180, 270, 365}; headline at the test-matching Δ. Paired bootstrap of per-fixture captured points against B0-stale (E16), 2000 resamples.
- Expectation written before running: **CB ties B0**, possibly worse on fixtures after a squad change.
- Diagnostic: per team, $\sum_i\hat Y_i$ against the realised team fantasy total (CB is not told that 11 play; a bias here shows whether it learned it).

## 9. Tests and gates

| id | test | fails if |
|---|---|---|
| T1 | causality: every token of an example is dated $< c$; recompute 200 examples from truncated raw data | any mismatch |
| T1b | candidate sets are the season pseudo-squads from `squads.py` | any difference |
| T1c | absence tokens obey §2.3 (after first appearance for $T$, within 365 d, before $c$) | any violation |
| T3 | overfit 50 fixtures: training loss → ≈ 0 | cannot overfit |
| T4 | permutation: shuffling candidates permutes outputs identically (Prop. CB-2) | any difference > 1e-6 |
| T4b | A/B swap: mean \|Ŷ(A,B) − Ŷ(B,A)\| on validation | reported, not blocking |
| T5 | determinism: two CPU runs give byte-identical CSVs | any diff |
| T7 | cold start: a candidate with 0 tokens runs and gets a finite output | error or NaN |

## 10. Known limitations
- Candidate sets in training are pseudo-squads (players who played at least once); test squads contain members who never play. The decoder's attention is normalised over the set, so a larger set with unfamiliar members is a stronger shift for CB than for a per-candidate model (Scrutiny S9).
- No guarantee of $\sum_i p_i = 11$ or of a keeper; whatever competition CB learns, it learns from data.
- A neural model in a "Traditional ML" track: B stays the fallback unless CB or an ADV rung wins by the rule in PIPELINE §4.

## 11. Changes from revision 2

| change | reason |
|---|---|
| NB (CT-WKV + capacity binding + BCE/L2 + τ) → CB (plain Transformer encoder + set decoder + one MSE head) | the user's decision: the baseline should ask the model to learn every mapping itself, so each structural idea is measured against a model that had to find it unaided |
| CT-WKV and capacity binding moved to ADV §A0 | they are structure (a decay law; an explicit capacity representation) |
| bench tokens → absence tokens with a 365-day window | the same raw fact with a simpler rule; closes rev. 2 open question 3 for CB |
| per-candidate decoder → set decoder over both squads | without it the caveman cannot even represent competition for places |

## 12. Open for scrutiny (close before implementing)
1. Absence-token rule: 365-day window after first appearance for $T$, or the season plus the previous season?
2. Decoder over both squads (proposed), own squad only, or none?
3. $L$ = 64 tokens: enough for p (recent absences) and μ (career form) together?
4. Put the realised points $3+\Phi$ in each token (proposed), or raw counts only (purer, but the model must then learn Φ itself)?
5. A/B swap augmentation (proposed) or an architecture that is symmetric by construction (relation embedding "same team / other team" inside attention)?
6. Candidate-set size shift (Limitation 1): accept, or add training candidates from the extended squad (`in_ext`) so sets are closer to 15+?
7. $\mathcal D_\Delta$ as PIPELINE D3, until the test file (O2).
