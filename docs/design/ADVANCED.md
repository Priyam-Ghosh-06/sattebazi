# Advanced architecture (ADV)

Status: design, revision 2.1 (2026-09-29, night). Revision 2.1 only moves material: the baseline became the caveman model (`BASELINE.md` rev. 3), so the CT-WKV encoder and the capacity-bound factorised decoder, formerly the baseline, are now ADV's first rungs (§A0). Nothing in A1–A6 changed. **Do not implement a component until its items in §12 are closed.**
Starts from the caveman baseline CB. Every component is a flag and enters through the ablation ladder (§10), so the gap between CB and each rung is attributable to that rung's structure.
Reference: Karlsson, Willbo, Hussain, Krishnan, Sontag, Johansson, *Using Time-Series Privileged Information for Provably Efficient Learning of Prediction Models*, AISTATS 2022 (arXiv:2110.14993), called **LuPTS** below (`references/`).

---

## 0. Map from ideas to components

| component | idea (THOUGHT_LOG) | what it is |
|---|---|---|
| A0a CT-WKV encoder | Entry 4 (history encoder that stores time) | memory that decays per appearance and per day; explicit state for A5 |
| A0b factorised decoder | Entry 5 (two problems: plays, points if plays); role–filler | separate $p$ and $\mu$ heads with capacity binding, BCE + L2, τ |
| **A3 choice layer** | Entry 1.1, Entry 5 (who plays; the backup-keeper counter; the ranker counter) | exact constrained subset-choice model over each squad; replaces revision 1's in-network sum-to-XI |
| A2 directional context | Entry 1.2 (setting is a different *direction* of information than opponent) | FiLM for global context, log-link offsets in rate heads, pairwise term for opponent |
| A1 squad attention | Entry 1.1, 1.3 (candidate set, conditional-mean synergy) | own-team self-attention and opponent cross-attention; **μ path first** |
| **A4 innings simulator** | Entry 1.5 (hierarchical experts), Φ's structure, within-match privileged information | the unit of simulation is the innings: team-level head, then allocation heads that respect every accounting identity; Φ applied exactly |
| A5 temporal privileged training | Entry 4 (privileged training on time) | month-by-month latent transition learned from matches inside the staleness gap |
| A6 distillation | LuPTS Theorem 2 | one-pass student trained on a λ-mix of labels and the A4/A5 teacher |

Dropped, with evidence: listwise and rank losses (E10 tie, E13 ListNet worse), orthogonality penalties (E12 tie; replaced by supervised specialisation in A4), LLM reranker (post-cutoff leakage), ArcFace (membership is relative to the squad, not a class). The knowledge-graph idea survives as shared event embeddings, identified through players who play in several leagues.

---

## A0. Structured encoder and factorised decoder (moved from BASELINE rev. 2)

CB learns everything from raw tokens with a generic Transformer and one MSE head. A0 replaces the two generic pieces with structure, one rung at a time: **A0a** a history encoder whose memory decays in real time, **A0b** a decoder that outputs $(p_i, \mu_i)$ separately with explicit capacities. CB + A0a + A0b is the old neural baseline NB, now read as structure added to CB.

### A0.1 Extra inputs
**Capacity vector** $c_i\in\{0,1\}^Q$, $Q = \{\text{keep}, \text{bowl}, \text{bat}\}$, as of $c$ (PIPELINE D1). An all-rounder has $c = e_{\text{bowl}} + e_{\text{bat}}$; a keeper-batter $e_{\text{keep}} + e_{\text{bat}}$; no history gives $c = 0$ and `unk = 1`. Overseas enters as a binary input here and as a constraint in A3. Fantasy roles stay output-only.

**Bench tokens** (A0a replaces CB's absence tokens with the causal rule): a bench token at team match $j$ is emitted only for players who appeared for that team at or before $j$ in that season, or in the previous season. Candidate sets stay the season pseudo-squad (T1b, Scrutiny S8).

### A0.2 A0a: continuous-time WKV encoder (CT-WKV)
Why: E4b fitted local-level Kalman memories with half-lives of 15–50 appearances, several memories together beat any single one (E4), histories are irregular in time, and A5 needs an explicit fixed-size state it can roll forward without observations. RMT and Block-Recurrent Transformers solve contexts longer than the attention window (not our problem); xLSTM's matrix memory is built for retrieval, we need smoothing.

For token $n$ with input $x_n$, day gap $\delta_n$, channel $c = 1..d$:

$$
a_{n,c} = \exp\!\big(-\lambda^{\text{app}}_c - \lambda^{\text{day}}_c\,\delta_n\big),\quad \lambda = \operatorname{softplus}(w),\qquad
k_n = W_kx_n,\ v_n = W_vx_n,\ r_n = \sigma(W_rx_n),
$$
$$
N_n = a_n\odot N_{n-1} + e^{k_n}\odot v_n,\qquad D_n = a_n\odot D_{n-1} + e^{k_n},\qquad o_n = r_n\odot N_n/D_n .
$$

Unrolled, $N_n/D_n = \sum_{j\le n}\omega_{nj}v_j/\sum_{j\le n}\omega_{nj}$ with $\omega_{nj} = \exp\big(k_j - \lambda^{\text{app}}(n-j) - \lambda^{\text{day}}(t_n - t_j)\big)$.

**Proposition 1 (B0 containment).** With $k\equiv$ const, $\lambda^{\text{day}} = 0$, $v_j = y_j$: $o_n/r_n$ is the normalised EWMA with half-life $\ln 2/\lambda^{\text{app}}$; a bank of channels reproduces B0's {3, 10, 30} bank. *Proof.* $e^k$ cancels between numerator and denominator. ∎

**Proposition 2 (staleness moves confidence, not the estimate).** With no token between $t_n$ and the query $t^\ast$: $N^\ast/D^\ast = N_n/D_n$ and $\log D^\ast = \log D_n - \lambda^{\text{day}}(t^\ast - t_n)$. *Proof.* Both are multiplied by $e^{-\lambda^{\text{day}}(t^\ast - t_n)}$. ∎ The decoder receives $[o^\ast, \log D^\ast]$: an estimate and an effective-sample-size signal that falls linearly with the gap (the analogue of a Kalman prediction step).

Kalman link: steady-state gain $K = 1 - e^{-\lambda}$. Initialise half-lives log-uniform over [2, 64] appearances and [30, 1000] days. Cold start: $o^\ast = 0$, $\log D^\ast = \log 10^{-4}$, plus a learned $e_{\text{cold}}$. Numerics in log-space with a running max (RWKV-v4). 2 layers [CT-WKV time-mix → channel-mix MLP], pre-LN, $d = 64$; truncated BPTT over the last 128 tokens (state exact, gradient truncated). Tests T2 (EWMA containment vs `pandas.ewm`, error < 1e-5) and T6 (Prop. 2 to 1e-6).

### A0.3 A0b: factorised, capacity-bound decoder
Candidate vector $u_i = \operatorname{MLP}([o_i^\ast, \log D_i^\ast, \log(1+\delta_i^\ast), \text{side}_i, \text{overseas}_i, \text{unk}_i, g_m]) + \text{unk}_i e_{\text{cold}}$, with $g_m$ the pre-toss fixture encoding (format, gender, team type, event, venue, opponent, month, overs, $\log(1+\Delta)$). No interaction between candidates at this rung.

**Capacity binding** $b_i = u_i\odot(\mathbf 1 + Rc_i)$, $R\in\mathbb R^{d\times Q}$. **Proposition 3.** For $c = e_q$: $b = u + u\odot r_q$ (Hadamard role–filler binding); for an all-rounder the two bindings superpose; for $c = 0$, $b = u$. *Proof.* Linearity of $Rc$ and distributivity of $\odot$. ∎ The diagonal is the rank-restricted compression of the tensor product $r\otimes f$.

Heads $s_i = w_p^\top\operatorname{MLP}_p(b_i)$ (appearance logit) and $\mu_i = \sigma_Yw_\mu^\top\operatorname{MLP}_\mu(b_i) + \bar y$. Loss $\frac1{|S|}\sum_{S}\operatorname{BCE}(\sigma(s_i), D_i) + \frac{1}{|P|}\sum_{P}((\mu_i - y_i)/\sigma_Y)^2$; minimisers $P(D=1\mid\cdot)$ and $E[Y\mid D=1,\cdot]$.

**Sum-to-XI** (post-hoc): per (fixture, team) solve $\sum_i\sigma(s_i - \tau) = n_{XI}$ by bisection; the I-projection of the independent model onto the expected-size constraint (an exponential tilt = a common shift of natural parameters). It cannot move probability within a role (Scrutiny S2); A3 replaces it.

---

## 1. The raw system with its privileged variables

For fixture $m$ the natural chain of random variables is

$$
\underbrace{X_1}_{\mathcal F_{\text{pre}}:\ \text{state at }c,\ \text{fixture, squads}}
\to \underbrace{X_2}_{\text{XI of both teams, toss}}
\to \underbrace{X_3}_{\text{innings totals, batting order, balls faced, overs bowled}}
\to \underbrace{X_4}_{\text{runs, 4s, 6s, outs, dots, wickets, catches, …}}
\to Y_i = \mathbb 1[i\in X](3 + \Phi_i(X_3, X_4)).
$$

Training observes the whole chain; the test observes only $X_1$. There are two kinds of privileged information:
- **Vertical (within the match):** $X_2, X_3, X_4$, all post-toss.
- **Horizontal (across the staleness gap):** a player's matches in $(c, \text{date}(m))$. They are unobservable at test time; in training they are observable after truncating history at $c = \text{date} - \Delta$.

### 1.1 What LuPTS does, and its price
LuPTS fits each transition $\hat f_t: X_t\mapsto X_{t+1}$ separately on privileged pairs, fits $\hat g: X_T\mapsto Y$, and predicts $\hat g\circ\hat f_{T-1}\circ\dots\circ\hat f_1$ from $X_1$.
- **Theorem 1** (linear-Gaussian, Markov): the estimator is unbiased and $\mathrm{MSE}_{\text{LuPTS}} = \mathrm{MSE}_{\text{OLS}} - E[\mathrm{Var}(\hat\theta_{\text{OLS}}\mid\hat\theta_{\text{LuPTS}})]$. This is Rao–Blackwellisation: never worse than ignoring the intermediate data.
- **Theorem 2**: distillation gives $\lambda\hat\theta_{\text{OLS}} + (1-\lambda)\hat\theta_{\text{LuPTS}}$, a bias–variance valve.
- **Theorem 3** (any function class): $R(\hat g\circ\hat f) \le R_{X_T}(\hat f) + R_Y(\hat g) + 2\sqrt{R_{X_T}R_Y}$. Roll-out error compounds with the horizon.
- **Price**: the guarantee needs Markovianity and linear-Gaussian well-specification.

### 1.2 Our setting against those assumptions
1. **Markov** fails as stated: skill in $X_1$ affects $X_4$ directly, not only through $X_3$. Fix: carry $X_1$ forward, $\tilde X_t = (X_t, X_1)$, which makes the chain Markov trivially while keeping the separate-fitting mechanism. Every head below conditions on the player states $h_i$.
2. **The last step is noiseless**: $\Phi$ is known and exact, so $R_Y = 0$ and Theorem 3 reduces to $R \le R_{X_T}(\hat f)$. All error lives in the dynamics, and any gain must come from the noisy transitions $X_1\to X_2\to X_3\to X_4$.
3. **Linear-Gaussian** fails (counts, thresholds). Theorem 1 does not transfer; we rely on Theorem 3 and on measurement. The report says so.
4. **Composing point predictions is biased**: $E[\Phi(X_4)] \ne \Phi(E[X_4])$ for the milestone, band and haul terms. The transitions therefore output **distributions**, and prediction uses the tower property by Monte Carlo through the exact Φ (§5.6). If the conditionals are correct this is unbiased; the only error is transition misspecification.
5. **B0 is already a two-step LuPTS**: $E[Y\mid X_1] = P(\text{played}\mid X_1)\,E[Y\mid\text{played}, X_1]$. The hindsight gap (0.691 → 0.743 captured/oracle at Δ = 0) bounds what better modelling of $X_2$ alone can add at Δ = 0.
6. **Horizontal axis**: the CT-WKV state is a fixed-size summary, so month-to-month Markovianity in state space is an approximation; one shared transition (Stat-LuPTS) is assumed apart from a season-phase input.

---

## 2. A3: the constrained choice layer

### 2.1 Model
For one team's candidates $S = \{1..n\}$ with utilities $u_i$ (from the decoder head) and capacity vectors $\tilde c_i = (1, c_{i,\text{keep}}, c_{i,\text{bowl}}, c_{i,\text{ovs}})$:

$$
P(X\mid S) = \frac{1}{Z_C(S)}\exp\Big(\sum_{i\in X}u_i + \sum_{q}\phi_q\big(n_q(X)\big)\Big)\,\mathbb 1\big[|X| = n_{XI}\big],\qquad n_q(X) = \sum_{i\in X}c_{iq}.
$$

$\phi_q$ is a learned table over counts. Hard constraints are $-\infty$ entries: overseas above the competition cap, where the cap is the maximum observed count in that competition, checked by gate G-p1 (zero violations in historical XIs). Soft preferences (about one keeper, about 5–7 bowling options) are finite entries. Keeper is soft on purpose, because two keepers do play together.

### 2.2 Exact normaliser by dynamic programming
State $s = (m, k, b, o)$: selected count $m \le n_{XI}$, keepers $k$, bowling options $b$, overseas $o$. The counters $k$ and $b$ saturate at 3 and 8 ("3 or more", "8 or more"); $o$ is cut at the cap. Let $\oplus$ be saturating addition. Forward push in log-space:

$$
\alpha_0(\mathbf 0) = 0,\qquad
\alpha_i(s') = \operatorname{logaddexp}\Big(\alpha_{i-1}(s'),\ \operatorname*{logsumexp}_{s:\ s\oplus\tilde c_i = s'}\big[\alpha_{i-1}(s) + u_i\big]\Big),
$$
$$
\log Z_C = \operatorname*{logsumexp}_{s:\ m = n_{XI}}\Big[\alpha_n(s) + \sum_q\phi_q(s_q)\Big].
$$

The push form keeps the recursion exact under saturation, where a state has more than one predecessor. Size: $12 \times 4 \times 9 \times 5 = 2{,}160$ states, so $O(n \cdot 2160)$ per team-match, about $4\times10^4$ operations for a 20-player squad. Batched over fixtures this is negligible next to the encoder.

### 2.3 Probabilities are gradients
**Proposition 4.** $\pi_i := P(i\in X\mid S) = \partial\log Z_C/\partial u_i$ and $P(n_q = k) = \partial\log Z_C/\partial\phi_q(k)$.
*Proof.* $Z_C = \sum_X \exp E(X)$ with $E$ linear in $u$ and in the table entries, and $\partial E(X)/\partial u_i = \mathbb 1[i\in X]$. So $\partial Z_C/\partial u_i = \sum_{X\ni i}\exp E(X)$; divide by $Z_C$. The same argument applies to $\phi_q(k)$ with indicator $\mathbb 1[n_q(X) = k]$. ∎

So one autograd backward pass through the DP gives every exact marginal. Revision 1's bisection and implicit-function-theorem gradient disappear.

**Loss and its gradients.**

$$
\mathcal L_{\text{choice}} = \log Z_C - \sum_{i\in X_{\text{obs}}}u_i - \sum_q\phi_q\big(n_q(X_{\text{obs}})\big),\qquad
\frac{\partial\mathcal L}{\partial u_i} = \pi_i - \mathbb 1[i\in X_{\text{obs}}],\qquad
\frac{\partial\mathcal L}{\partial\phi_q(k)} = P(n_q = k) - \mathbb 1[n_q(X_{\text{obs}}) = k].
$$

The potentials are therefore fitted by **moment matching of count distributions**. $\nabla^2\log Z_C = \operatorname{Cov}(\mathbb 1[i\in X], \dots)\succeq 0$, so $\mathcal L_{\text{choice}}$ is convex in $(u, \phi)$ (not in the network weights that produce $u$).

### 2.4 The backup-keeper case, exactly
**Proposition 5.** Let the only keepers be $W_1, W_2$, let $\phi_{\text{keep}}$ enforce exactly one keeper ($\phi_{\text{keep}}(1) = 0$, else $-\infty$), and let there be no other potentials. Then
(a) with both present, $\pi_{W_2} = \sigma(u_{W_2} - u_{W_1})$;
(b) with $W_1$ absent, $\pi_{W_2} = 1$ for every $u_{W_2}$.
*Proof.* Each feasible XI holds exactly one keeper plus $n_{XI}-1$ non-keepers from $N$. So $Z_C = (e^{u_{W_1}} + e^{u_{W_2}})\,e_{n_{XI}-1}(e^{u_N})$, where $e_k$ is the elementary symmetric polynomial. The XIs containing $W_2$ contribute $e^{u_{W_2}}e_{n_{XI}-1}(e^{u_N})$; divide. Without $W_1$, $Z_C = e^{u_{W_2}}e_{n_{XI}-1}(e^{u_N})$ and every feasible XI contains $W_2$. ∎

*Consequence for learning.* When $W_2$ is benched behind $W_1$, his gradient is $-\sigma(u_{W_2} - u_{W_1})$, which vanishes as the gap grows. Only the **difference** is learned, and his absolute standing is not pushed to $-\infty$. Independent BCE instead pushes $s_{W_2}\to-\infty$ with gradient $-\sigma(s_{W_2})$. With soft potentials the statement holds approximately. This is the formal answer to Scrutiny S2–S4.

### 2.5 Identifiability (gauge)
- A common shift $u_i\mapsto u_i + a$ within a team-match leaves $\mathcal L$ invariant, because $|X|$ is fixed. $\pi$ is unaffected, so no action is needed.
- $\phi_q(k)\mapsto\phi_q(k) + a_q + b_qk$ together with $u_i\mapsto u_i - b_qc_{iq}$ leaves every energy difference invariant. The tables are identified only up to an affine term. **Gauge fix:** set $\phi_q(k^{\text{ref}}_q) = \phi_q(k^{\text{ref}}_q + 1) = 0$ with reference counts keep 1, 2; bowl 5, 6; overseas 3, 4. That makes the tables interpretable and comparable across competitions (test T16).

### 2.6 Sampling and relation to τ
- **Exact joint samples** of the XI by backward sampling through the stored $\alpha$ tables (test T17). A4 needs these.
- **Relation to A0b.** With no potentials, $\pi_i = w_ie_{n_{XI}-1}(w_{-i})/e_{n_{XI}}(w)$ with $w = e^u$, the conditional-Bernoulli marginal. A0b's $\sigma(s_i-\tau)$ is its independent-Bernoulli (Poisson-design) approximation, whose error is of order $1/d$ with $d = \sum_i\pi_i(1-\pi_i)$. That is largest when most of the XI is locked in, which is our case.
- **Ranker limit.** Scaling $(u,\phi)$ by $1/T$ and letting $T\to0$ concentrates $P$ on the maximum-energy XI, the within-capacity ranker (Scrutiny S4). The temperature is absorbed into the scale of $u$ and fitted by the likelihood.
- **XI size.** $n_{XI}$ = 11, or 12 in impact-sub competitions (O4).

---

## 3. A2: context by direction
- **Global context** (format, venue, event, month, scheduled overs) conditions every candidate multiplicatively: $h_i\leftarrow\gamma(g_m)\odot h_i + \beta(g_m)$ (FiLM).
- In A4's rate heads it enters as a **log-link offset**, $\log\lambda = f(h_i) + b_{\text{venue}} + b_{\text{event}}$: additive on the log scale, multiplicative on rates. That is the formal version of "a flat pitch scales everyone's scoring". Venue and event offsets are sum-to-zero (identifiability).
- **Opponent** enters only pairwise, through A1's cross-attention.
- This is the decomposition $\log E[\text{runs}] = a_i + b_{\text{opp}} + c_{\text{venue}}$ from Entry 1. League strength is identified through the player–league graph (players who move between leagues).

## 4. A1: squad attention
Two layers of $h_i \leftarrow h_i + \operatorname{SelfAttn}_{\text{own team}}(h)_i + \operatorname{CrossAttn}_{i\to\text{opp. squad}}(h)_i$, then a channel-mix MLP. The layer is permutation-equivariant over candidates (the right symmetry for a set).

**Routing.** A1 feeds the **μ path first**. μ currently does not see the squad: a missing regular opener pushes the No. 3 up the order, which is conditional-mean synergy. Correlation-type synergy is irrelevant to the expected-optimal XI by linearity (Entry 1).

**Why not the p path by default.** A3 already represents competition for slots structurally. If attention makes $u_i$ depend on who else is in the squad, a backup who has never appeared without the first-choice player receives an input never seen in training when that player is absent. The hard case stays right, because $\phi_{\text{keep}}$ forces a keeper. Fringe players in other capacities depend on attention's behaviour off-distribution. A1 on the $p$ path is therefore its own ladder rung, with a pre-registered expectation of a tie or worse.

---

## 5. A4: the innings simulator (vertical privileged information)

Revision 1 predicted each player's opportunity independently. That allows six bowlers at four overs each in a 20-over innings, or two batters at No. 3. Φ's economy and strike-rate bands (2-over and 10-ball minimums) then apply to impossible opportunities, and fantasy XIs stacked from one team inherit the excess. Revision 2 makes the **innings** the unit of simulation: a team-level head, then allocation heads whose supports satisfy every accounting identity by construction.

### 5.1 Order of sampling for one fixture
1. **XIs**: one exact joint sample per team from A3 (§2.6).
2. **Toss / batting first**: $\text{BF}\sim\text{Bernoulli}(\pi_{\text{bf}})$, learned from team and venue history, with 0.5 as the ablation (D7).
3. **Innings 1** (A bats, B bowls), then **innings 2** conditioned on innings-1 runs (the target). Each innings runs §5.2–5.5.

### 5.2 Team-level innings head
Inputs: pooled states of A's sampled XI (batting) and B's sampled XI (bowling), $g_m$, innings index, and the target in innings 2. Pooling is attention over the sampled XI.
- Wickets $W\in\{0..10\}$: categorical.
- Legal balls $T \mid W \in\{0..6\,o_{\text{sched}}\}$: discretised Beta-binomial; all-out innings end early.
- Dismissal kinds $\kappa\mid W\sim\text{Multinomial}(W;\ \text{bowled}, \text{lbw}, \text{caught}, \text{stumped}, \text{hit wicket}, \text{run out}, \text{other})$.
- Extras (wides, no-balls): Poisson with log-link offsets.

### 5.3 Batting side (team A)
- **Order**: Plackett–Luce over the sampled XI,
  $$P(\sigma\mid X_A) = \prod_{k=1}^{|X_A|}\frac{e^{v_{\sigma(k)}}}{\sum_{j\ge k}e^{v_{\sigma(j)}}},\qquad v_i = f_v(h_i).$$
  The batting order is a permutation, so positions are distinct by construction. This closes revision 1's categorical-vs-ordinal question.
- **Batters used**: $K = \min(11, W+2)$; positions $> K$ are DNB.
- **Balls faced**: $(B_1..B_K)\sim\text{DirMult}\big(T;\ \alpha_1..\alpha_K\big)$ with $\alpha_k = \exp f_B(h_{\sigma(k)}, k)$. So $\sum_kB_k = T$ exactly, and the position enters $\alpha$.
- **Who is out**: exactly $W$ of the $K$ batters are dismissed. This is a conditional-Bernoulli draw with scores $f_d(h_{\sigma(k)}, B_k)$, computed with A3's DP restricted to the count state.
- **Runs**: $F_k\sim\text{Bin}(B_k, q_4)$, $S_k\sim\text{Bin}(B_k - F_k, q_6)$, other runs $\sim\text{NegBin}(\text{mean} = \rho\,(B_k - F_k - S_k))$. So runs $= 4F + 6S + R_{\text{other}}$, and the boundary, run and milestone terms are mutually consistent. Duck = dismissed ∧ runs = 0.

### 5.4 Bowling side (team B)
- **Overs**: over-by-over sequential allocation among B's sampled XI, with $P(\text{bowler } j \text{ bowls over } t)\propto e^{\omega_j}\,\mathbb 1[O_j < \lceil o_{\text{sched}}/5\rceil]\,\mathbb 1[j\ne\text{bowler of over } t-1]$ and $\omega_j = f_\omega(h_j)$ (bowling-option capacity raises it). Caps and the no-consecutive-overs law hold in every sample. The number of overs is $\lceil T/6\rceil$; the last is partial when the innings ends early.
- **Runs conceded**: $(R_j)\sim\text{DirMult}\big(\text{runs off the bat} + \text{wides} + \text{no-balls};\ \beta_jO_j\big)$. That equals the batting side's total minus byes and leg-byes, exactly.
- **Dots**: $\text{Bin}(\text{legal balls}_j,\ q_{\text{dot}}(h_j, R_j/\text{balls}_j))$. **Maidens**: $\text{Bin}(\text{complete overs}_j,\ q_m(h_j, R_j/O_j))$. Both are conditioned on the bowler's economy; the dot/maiden joint is an approximation (§11).
- **Wickets credited**: each of the $\kappa_{\text{bowled}} + \kappa_{\text{lbw}} + \kappa_{\text{caught}} + \kappa_{\text{stumped}} + \kappa_{\text{hw}}$ credited wickets is assigned to a bowler with probability $\propto O_je^{\psi_j}$. Bowled and lbw carry the +5 bonus.

### 5.5 Fielding (team B)
- Each caught dismissal goes to a catcher in B's XI with probability $\propto e^{\zeta_i}$ (keeper capacity raises it; catcher = bowler gives caught-and-bowled).
- Stumpings go to B's keeper(s).
- Run-outs: direct or indirect by Bernoulli, fielders $\propto e^{\zeta_i}$ (matches the Φ heuristic in DECISIONS.md).

### 5.6 From samples to scores
$\Phi_i$ is computed exactly by `src/points.py` from each trajectory's per-player counts. **Rao–Blackwellised estimator**: given the structural draw $Z$ (XIs, toss, $T$, $W$, $\kappa$, order, $K$, the out-set, over allocation, $B$), the linear parts of Φ (runs, boundaries, participation, wickets × 25, catches, dots) are replaced by closed-form conditional means. Only the nonlinear parts (milestones, SR and economy bands, hauls, duck, 3-catch bonus, maidens) are averaged over samples. $\operatorname{Var}(E[g\mid Z])\le\operatorname{Var}(g)$, so this never adds variance. $K = 4096$ trajectories per fixture with common random numbers (seed = hash(fixture)). The same trajectories give $E[Y_i]$ for the ILP, $\operatorname{median}(Y_i)$ for the MAE report, and quantiles for uncertainty plots.

### 5.7 Training A4
Teacher forcing on the true upstream values (true XIs, toss, $T$, $W$, $\kappa$, batting order, over sequence): LuPTS's separate fitting. Every allocation law has a tractable likelihood: Plackett–Luce, Dirichlet-multinomial, conditional Bernoulli (by DP), a product of masked softmaxes for overs, multinomials for credits. Loss $= \sum_{\text{heads}}\text{NLL}$, each divided by its value under the training-set marginal so heads are comparable. These are proper scoring rules, so each head is pushed toward the true conditional.

**The hierarchy is the user's hierarchical mixture of experts.** Each expert (innings, order, balls, overs, runs, credits, fielding) is identified by **its own supervised target** rather than by an orthogonality penalty, which cannot identify a decomposition (E12).

---

## 6. A5: temporal privileged training (horizontal privileged information)

Player state $z = [\,o, \log D\,]\in\mathbb R^{2d}$ is the encoder output (A0.2). Monthly grid τ = 30 days.
- **Privileged sequence**: for a training fixture with $\Delta\sim\mathcal D_\Delta$ and $c = \text{date} - \Delta$, compute $z(c), z(c+\tau), \dots, z(c+K\tau)$ by running the frozen encoder on the *actual* matches in the gap, which are observable only in training.
- **Transition**: $\hat z_{k+1} = z_k + \operatorname{MLP}_g([\,z_k,\ \text{season-phase}(c + k\tau),\ \text{gender},\ \text{team\_type}\,])$, one shared $g$ (Stat-LuPTS). `--nonstationary-g` gives one $g$ per calendar month.
- **Loss**: $\sum_k\|\hat z_{k+1}(z_k) - \operatorname{sg}(z_{k+1})\|^2$, one step at a time with teacher forcing (Algorithm 1's separate fits). **Collapse guard**: stop-gradient targets **and** a frozen encoder, so the constant-state solution is not available (test T12).
- **Test**: $z^\ast = g^{\lceil\Delta/\tau\rceil}(z(c))$, then the decoder.
- **Horizon**: at $\Delta\ge273$ days this is ≥ 10 steps, where Theorem 3's compounding bites hardest. The direct comparator (the paper's "OLS") is A0b's decoder on $z(c)$ with Δ as input; the staleness grid shows which one wins.

## 7. A6: LuPTS distillation
Student $y^{\text{dir}}_i = \operatorname{MLP}(h_i)$, one forward pass, no roll-out:

$$
\mathcal L_{\text{dist}} = \lambda\,(y^{\text{dir}}_i - y_i)^2 + (1-\lambda)\big(y^{\text{dir}}_i - \operatorname{sg}(\hat E_{\text{roll}}[Y_i])\big)^2,\qquad\lambda\in\{0, .25, .5, .75, 1\},
$$

with λ chosen on validation. The final predictor (student or roll-out) is chosen on validation captured.

---

## 8. Training schedule

| phase | trains | data | loss | notes |
|---|---|---|---|---|
| P1 | encoder, A1, A2, A3 (utilities and φ), A4 heads | Δ-augmented, as CB | $\mathcal L_{\text{choice}}$ + ΣNLL(A4) + L2 on μ-path | optimiser as CB, peak lr 2e-3 |
| P2 | $g$ only, encoder frozen | privileged monthly state pairs from fixtures with Δ > 0 | one-step latent MSE | lr 1e-3, ≤ 10 epochs |
| P3 | A1–A4 fine-tuned on rolled-forward $z^\ast$ (encoder and $g$ frozen) | Δ-augmented | as P1 | lr 3e-4; reduces the Theorem 3 mismatch |
| P4 | A6 student | as P3 | $\mathcal L_{\text{dist}}$ | λ grid |

Folds, validation, EMA, determinism and stopping as in BASELINE §7. Checkpoints are selected on validation captured/oracle at the test-matching Δ. Sweeps run on Kaggle; the final model is trained on CPU, deterministically.

## 9. Inference (`src/evaluate.py --arch adv`)
History ≤ 2025-12-31 → encoder → $z(c)$ → A5 roll-forward to the fixture date → A1/A2 over both squads → A3 utilities and potentials → A4 trajectories (XIs sampled from A3, Rao–Blackwellised) or the A6 student → $E[Y_i]$ for the ILP, median for the MAE report → CSV.

---

## 10. Tests, ablation ladder, decision rule

### 10.1 Tests (on top of BASELINE T1–T7 and A0's T2, T6)
| id | test | fails if |
|---|---|---|
| T7 | Φ identity: feeding true counts through the roll-out's Φ reproduces stored Φ on 1,000 rows | any mismatch |
| T8 | roll-out unbiasedness on a synthetic chain with known parameters | MC mean outside 3 SE of the analytic mean |
| T9 | joint calibration: PIT histogram of total $Y$ per role on validation | KS p < 0.01 (reported, not blocking) |
| T11 | privileged isolation: zeroing gap-window tokens leaves the $X_1$ path unchanged | any change |
| T12 | no collapse: across-player variance of $z$ after P2 ≥ 50% of before | collapsed |
| T13 | Rao–Blackwell: RB estimator variance across seeds ≤ plain MC | larger |
| T14 | A3 marginals equal brute-force enumeration on random squads of 14 ($\binom{14}{11} = 364$ subsets), with potentials | error > 1e-6 |
| T15 | accounting identities in $10^4$ simulated innings: $\sum B = T$ = legal balls bowled; overs ≤ cap; no consecutive overs; dismissed batters = $W$ = credited + run-outs + other; catches = $\kappa_{\text{caught}}$ | any violation |
| T16 | gauge: φ tables satisfy the reference-count constraints; $\pi$ unchanged by an affine re-gauge | violated |
| T17 | exact sampler: inclusion frequencies of $10^5$ backward samples lie within 3 SE of $\pi$ | outside |
| G-p1 | zero overseas-cap violations in historical XIs per competition (validates nationality inference) | any |
| G-p2 | log-loss on fixtures after a squad change (a regular missing, a new arrival) is reported separately | (reported) |

### 10.2 Ladder with pre-registered expectations
Each rung is compared with the previous rung and with B0-stale by paired bootstrap on full 2024 and 2025 folds, at Δ ∈ {0, 90, 180, 270, 365}.

| rung | expectation (written before running) |
|---|---|
| CB | ties B0 |
| + A0a (CT-WKV encoder) | tie or small gain; larger at large Δ if Prop. 2's confidence signal helps |
| + A0b (p/μ heads, capacities, τ) = old NB | tie; better p log-loss |
| + A3 (choice layer, capacities) | better log-loss on the squad-change segment (G-p2); overall captured +0 to +2 |
| + A2 | ≈ E14's +2.6 if the plain embeddings missed it; else tie |
| + A1 → μ | lower μ error on fixtures missing a regular; captured small gain |
| + A1 → p | tie or worse (off-distribution utilities, §4) |
| + A4 | better calibrated $E[Y]$ for fringe candidates; E18's team-total bias removed; captured +0 to +3; MAE better via the roll-out median |
| + A5 | 0 at Δ = 0 by construction; grows with Δ if the monthly dynamics are well estimated, shrinks or reverses if biased |
| + A6 | never worse than the better of direct and roll-out at the tuned λ (guaranteed only in the linear case) |

**Adoption rule**: a rung is kept if its CI against the previous rung is not significantly negative at the test-matching Δ and it improves captured or validation NLL. The submission is the best of {B, CB, ADV-best} on 2024, confirmed on 2025. If nothing beats B, B is submitted and this ladder is the ablation study.

---

## 11. Honest risks
- None of LuPTS's guarantees transfer to this nonlinear, non-Gaussian, state-augmented setting. The claim is "a principled estimator with a measured bias–variance trade-off", not "provably better".
- A4 approximations: dots and maidens are conditioned on economy but not jointly consistent with the ball-by-ball sequence; batting partnerships are summarised by a Dirichlet-multinomial split rather than simulated ball by ball. T9 measures the effect on the tails, which is where milestone and haul bonuses live.
- A3 depends on capacity labels. A mislabelled regular keeper makes an XI look keeperless, which softens $\phi_{\text{keep}}(0)$. Gate: fraction of historical XIs with 0 inferred keepers ≈ 0.
- A5 depends on the test Δ; if the organisers allow post-cutoff inputs (O3), A5 is moot.
- Impact-sub XIs of 12: the batting side uses at most 11, so the simulator needs a rule for which 11 bat (open, §12).

## 12. Open for scrutiny (close before implementing)
1. **A3**: which potentials (keeper, bowling options, overseas, batting depth?), and global vs per-competition tables (PIPELINE D2).
2. **A3**: capacity label rules (PIPELINE D1).
3. **A1**: keep the $p$-path rung, or drop it (D5).
4. **A4**: teacher forcing only, or scheduled sampling on $X_3$ (D6).
5. **A4**: batting-first probability learned or 0.5 (D7).
6. **A4**: impact-sub fixtures: which 11 of 12 bat and bowl, and whether the substituted-out player's events count.
7. **A5**: monthly grid vs one step per team match (D8).
8. **P3**: fine-tune the encoder end-to-end (D9; lean no).
9. **K** = 4096: check against the distribution of $|\text{score}_{11} - \text{score}_{12}|$ at the ILP boundary (D10).
10. **A4 balls faced vs Φ** (found 2026-09-29 night while building the pipeline dashboard): §5.3 sets $\sum_kB_k = T$ = legal balls, but Φ's balls faced also counts no-balls (ingest), so simulated SR denominators are short by the no-balls each batter faced. Proposal: allocate the simulated no-balls to batters in proportion to $B_k$ and add them to $B_k$ before Φ.

## 13. Changes from revision 1
| change | reason |
|---|---|
| A3 sum-to-XI → exact constrained choice layer with capacity potentials | Scrutiny S2–S7; probabilities by autograd (Prop. 4); backup-keeper case exact (Prop. 5) |
| A4 per-player heads → innings simulator with allocation heads | accounting identities (overs, balls, wickets) held by construction; batting order as a permutation |
| A1 routed to μ first | competition is now structural in A3; off-distribution risk on the p path |
| gauge fixing, exact sampler, tests T14–T17 | identifiability and correctness of A3 |
| $\mathcal D_\Delta$ to 450 days; K 512 → 4096; Kaggle sweeps, CPU final | test Δ likely ≥ 273; compute is not the constraint |
