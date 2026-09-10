# AlphaZero Trader — Research Track: World Model & The Exploitation Problem

> **Branch:** `research/track-2-world-model-fix` — this is the **public research track**
> **Status:** 🔬 Research Archive + Open Roadmap — world model code preserved, exploitation defenses proposed
> **Companion (private):** `deploy/track-1-long-flat` — production deployment with a C&C regime filter (not published)

An attempt to adapt **AlphaZero** to financial markets — treating trading as a single-player “market game” optimised for **Sharpe ratio** with a **Transformer** backbone and **MCTS as training oracle**. This README tells the full story from the git history: what worked, what failed, and why the project split into two tracks.

---

## TL;DR

- **We built it:** AlphaZero → trading, Transformer policy/value, MCTS-oracle training on historical ticks, Sharpe-based rewards, realistic friction (fees, idling penalty, drawdown stop).
- **Wall #1:** Regime blindness — the model could not handle bear/bull/range transitions.
- **Two bets to fix it:**
  1. **C&C (Command & Control) layer** — production pragmatism, ADX-gated regime filter outside the RL loop.
  2. **World Model** — architectural bet: a Conv1D+LSTM transition model so MCTS can plan beyond history.
- **Wall #2:** The world model was exploited. MCTS found 30–50% fantasy returns per episode through compounding prediction error. The policy learned the fantasy, failed on all real regimes.
- **External confirmation:** ~2 months after we paused, a paper formalised the same failure as the **exploitation problem in world models** — [arxiv.org/pdf/2605.15960](https://arxiv.org/pdf/2605.15960).
- **Decision:** Split the repo. This public branch is the **research track** (world model preserved, C&C stripped). The deploy track stays private.

If you are here to reproduce, extend, or fix the world model — you are in the right place.

---

## Timeline — From Git Log

| Date | Commit | What happened |
|---|---|---|
| **2026-03-30** | `901cef4` basic alphazero trader | Initial port: AlphaZero loop, `MCTS.py`, `game.py`, `model.py` (Transformer), CCXT scaffolding |
| **2026-03-31** | `eb23fd3` fixed basic version | `config.yaml` introduced (60-window, Sharpe target), ResNet→Transformer stabilisation |
| **2026-04-07** | `85e0f9c` phase 1 | Self-play → retrain → tournament loop locked; holding penalty ↑, MCTS sims 50→100 |
| **2026-04-08** | `61b2d07` 5 phases are implemented | Reward = `tanh(Sharpe)`, batched MCTS (batch 16), LR decay — the “real” training loop |
| **2026-04-08** | `5341f07` implementing multi architecture | Ablation: `transformer` vs `resnet` vs `causal_cnn_attn` |
| **2026-04-09** | `ebee1a3` fixing 1 player playing | Correct single-player handling, first `backtest.py` |
| **2026-04-09** | `64c440f` backtest on bear market | Bear-2022 stress test, `baseline_ema.py`, `profile_pipeline.py` — first regime split |
| **2026-05-05** | `39ad362` base architecture | Major overhaul: `data/stable/{training,val,backtest}`, `paper_trader.py`/`live_game.py`/`train.py` separation, `config.yaml` → `window 60`, `memory 100k`, `warmup` |
| **2026-05-06** | `3ec41e5` features fix for regime blindness | Added `log_return`, `rsi`, `atr`, `vol_10/30`, `rel_high_low`, `vol_delta` + portfolio state — **regime fix attempt that hit the wall** |
| **2026-07-17** | `500ade2` snapshot before track split | **World model era** snapshot: Conv1D+LSTM `world_model.py`, `transition_memory.py`, `train_world_model.py`, `validate_world_model.py`, `diagnose_mcts_sensitivity.py`, `inspect_synthetic_episodes.py`, `recalibrate_value_head.py` + **Fair Mode** `backtest_trades_v0018/v0019` on 4 regimes (all fail) |
| **2026-07-17** | `86dfcad` / `ba774ea` track split | `deploy/track-1-long-flat` (strip WM, add C&C filter) vs `research/track-2-world-model-fix` (preserve WM, add exploitation roadmap) |

This branch is `500ade2` + the research roadmap — **no C&C code**.

---

## Architecture at Snapshot (v18/v19)

```
[Historical OHLCV] ──► [data_manager.py] ──► [TradingGame] (game.py)
                                                │
                                        [MCTS Oracle] (MCTS.py, 600 sims, batch 16)
                                                │
                                         [Transformer 64d/4-head/2-layer] (model.py)
                                                ├─► Value Head (tanh Sharpe)
                                                └─► Policy Head (Flat/Long/Short)
                                                │
                                         [World Model] Conv1D(64×2) → LSTM(128×2) → Dense
                                                │  input: (window 60 × 19) + action one-hot
                                                │  output: predicted next state
                                                └─► used inside MCTS expansion (agent.py)
                                                              │
                                                        [TransitionMemory] (transition_memory.py)
                                                              │
                                                        retrained every 2 iterations (main.py)
```

**Training cycle (`main.py:99-179`):**
1. **Self-play** (`funcs.py:playMatches`) — random slices of training data, MCTS labels optimal moves.
2. **WM Retrain** every 2 iterations — sample 10k transitions, 5 epochs, batch 128.
3. **Transformer Retrain** — replay buffer `memory.ltmemory` ≥ 10% capacity.
4. **Tournament** — 3 regimes (`bear.csv`, `bull.csv`, `range.csv`), promote if `avg_sharpe > 0.02` and ≥2/3 regimes positive.

**State (`game.py`):** 60× (7 market features + 5 portfolio features), reward = `tanh(Sharpe/…)` with 1–5 bps fees, `drawdown_threshold 0.15`, `idling_penalty 0.00005`.

---

## The Two Bets After the Wall

### Bet 1 — C&C Layer (private, `deploy/track-1-long-flat`)

Production pragmatism: keep the proven historical-MCTS policy and gate risk **outside** the network with an ADX regime filter. “Many trading firms actually do this.” Not published here — see the other branch.

### Bet 2 — World Model (this branch)

Enable MCTS to roll **into predicted futures** rather than only historical ticks. Mechanically sound: gates passed, value head separated, MCTS divergence behaved as expected. Green lights everywhere — until backtest.

---

## Why the World Model Failed — The Exploitation Problem

**Fair Mode backtest (`backtest.py`, `--fair` flag: no MCTS lookahead, pure policy inference):**

| Model | Bear 2022 | Bull 2025 | Chop 2023 | Range 2024 |
|---|---|---|---|---|
| v18 | ❌ collapse | ❌ collapse | ❌ collapse | ❌ collapse |
| v19 | ❌ collapse | ❌ collapse | ❌ collapse | ❌ collapse |

Detail traces in `backtest_trades_v0018_*.csv` / `backtest_trades_v0019_*.csv`.

**Root cause — compounding error → planning exploit:**

```
Real move:      ±0.3% per tick
WM residual:    ±0.8% per tick (even when “good”)
20-step MCTS:   up to ±16% accumulated drift

→ MCTS(600 sims) hunts the error surface, finds fantasy paths = 30–50% episode return
→ Policy gradient treats fantasy as signal, calibrates value head to it
→ Policy converges to “play the hallucinated market,” useless on real data
```

This is **not** a bug in `agent.py:187-224` batched WM expansion or `world_model.py` — it is a fundamental planning-under-imperfect-model pathology. Diagnostic evidence is in `diagnose_mcts_sensitivity.py`, `inspect_synthetic_episodes.py`, `validate_world_model.py`, and `diagnostic_policy_regime.py`.

Two months later, the paper at **https://arxiv.org/pdf/2605.15960** formalised the same mechanism as *world-model exploitation* — an independent confirmation that the wall we hit is a known open problem.

---

## What This (Public) Branch Contains

**Keeps — everything needed to reproduce/extend the research:**

| File | Role |
|---|---|
| `world_model.py` | Conv1D+LSTM transition predictor (`TransitionModel`) |
| `transition_memory.py` | (s,a,s′) replay buffer |
| `train_world_model.py` | Standalone WM trainer |
| `validate_world_model.py` | WM error analysis |
| `collect_transitions.py` | Transition collection |
| `agent.py` | Batched MCTS expansion with `world_model=None` slot and `wm_depth ≥5` cap + WM backfill (`predict_batch` → `takeActionWM_FromPrediction`) |
| `main.py` | Self-play + WM retrain every 2 iterations |
| `game.py` | `takeActionWM` / `takeActionWM_FromPrediction` + `wm_depth` tracking |
| `MCTS.py` | Batched MCTS oracle |
| `model.py` | Transformer encoder + value/policy heads |
| `funcs.py`, `memory.py`, `data_manager.py` | Orchestration |
| `backtest.py` + `backtest_trades_v0018/19_*.csv` | Fair Mode evidence (8 traces, 4 regimes × 2 versions) |
| `diagnose_mcts_sensitivity.py`, `inspect_synthetic_episodes.py`, `recalibrate_value_head.py`, `diagnostic_policy_regime.py` | Failure diagnostics |
| `config.yaml` | `use_world_model: true`, 600 sims, batch 16 — the exact failing config |

**Removed for publication (C&C / deploy layer — stays private):**

- `c2_server/` (FastAPI C2 backend), `c2_ui/` (Next.js dashboard), `paper_trader.py`, `live_game.py`, `README_C2.md`

No credentials are published — see Security note below.

---

## Quick Start (Research)

```bash
python3 -m venv venv && source venv/bin/activate
pip install tensorflow keras ccxt pandas pandas_ta PyYAML matplotlib psutil natsort

# 1. Fetch data (creates data/stable/{training,val,backtest})
python3 data_manager.py --symbol BTC/USDT --timeframe 1h --train --limit 15000

# 2. (Optional) Train the world model standalone
python3 train_world_model.py

# 3. Full self-play loop with WM (the failing regime — for reproduction)
#    Uses config.yaml as-is (use_world_model: true)
export PYTHONPATH=$PYTHONPATH:.
python3 main.py

# 4. Validate WM in isolation
python3 validate_world_model.py

# 5. Fair Mode backtest (no lookahead — the ground-truth gate)
python3 backtest.py --model run/models/BTC_USDT_1h_v0019.keras --data data/backtest/bear_2022.csv
# Trade trace: backtest_trades_v*.csv
```

Switch granularity via `config.yaml:trading.timeframe` — retune `fee`, `idling_penalty`, and reward scaling per `README` history when moving 1h ↔ 5m.

---

## Roadmap — Fixing Exploitation (Contributions Welcome)

The paper at `2605.15960` gives the formalism; below are concrete interventions mapped to this codebase (from `research/track-2` roadmap at split time):

### A — Constrain how MCTS uses the WM (surgical, no arch change)
- [ ] **Rollout return ceiling** — `np.clip(v, -2*sigma, 2*sigma)` in `agent.py` WM backfill loop — *highest expected impact*
- [ ] **WM depth limit** — `wm_depth` 5 → 2 (error grows geometrically)
- [ ] **WM confidence** — MC Dropout / ensemble variance → discount `v` by accumulated uncertainty

### B — Fix the training signal
- [ ] **KL regularisation** — anchor `π(·|WM-state)` to `π(·|real-state)`
- [ ] **Value-head reality anchoring** — blend MCTS-on-WM value with NN-on-real-next-state value
- [ ] **Split replay** — `Memory` (history) + `TransitionMemory` (WM), train 75/25 blend

### C — Make the WM less exploitable
- [ ] **Residual WM** — predict `Δstate`, not `state` (slower compounding)
- [ ] **Ensemble WM (×3) + pessimism** — `min_ensemble(v)` for expansion
- [ ] **Error feedback loop** — per-step WM error → down-weight rollout confidence

**Suggested first experiment (A):** ceiling rollout return to `2× historical_sigma` (~0.06 for 1h BTC) in `agent.py:219-224` WM loop. Gate: Fair Mode drawdown no longer hits the 15% stop on `bear_2022.csv`.

Log results in the table below and open a PR against this branch.

### Experiment Log

| Date | Experiment | Hypothesis | Result |
|---|---|---|---|
| 2026-07-17 | Fair Mode v18 & v19 | WM causes exploitation | ✅ CONFIRMED — total failure, all regimes |
| — | Rollout ceiling (A) | Cap fantasy returns | — |
| — | KL regularisation (B) | Anchor policy to real data | — |
| — | Ensemble WM (C) | Pessimistic rollouts | — |

---

## Project Structure

```
AZT/
├── game.py                  # Market environment, GameState, takeActionWM
├── model.py                 # Transformer encoder, value/policy heads
├── MCTS.py                  # Batched MCTS oracle
├── agent.py                 # Agent ↔ MCTS bridge (WM-aware)
├── world_model.py           # TransitionModel (Conv1D+LSTM)
├── transition_memory.py     # (s,a,s′) buffer
├── main.py                  # Training loop (self-play → retrain → tournament)
├── funcs.py                 # playMatches / playMatchesSingle
├── memory.py                # Policy replay buffer
├── data_manager.py          # CCXT fetcher, regime splits
├── backtest.py              # Fair Mode backtester
├── train_world_model.py     # Standalone WM training
├── validate_world_model.py  # WM diagnostics
├── config.yaml              # All tunable params (use_world_model: true here)
├── backtest_trades_v0018/19_*.csv  # Fair Mode evidence
└── run/                     # .gitignored: models (.keras), memory (.p)
```

---

## Security & Privacy — What Was Cleaned

Before publication:

- **No `.env` ever committed** — git history contains zero `.env` blobs; `.env` is `.gitignored:115`. Only `.env.example` (placeholders) is tracked.
- **No hardcoded secrets** — `git grep BINANCE_API` hits only `os.getenv` + docs placeholders; scan of all branches finds zero `ghp_` or real keys.
- **Remote URL sanitised** — a GitHub PAT (`ghp_…`) was embedded in `remote origin` as `https://ghp_…@github.com/…`. **Removed** → `https://github.com/clusterO/alphazerotrader.git`. **If that PAT is still valid, revoke it immediately** (GitHub → Settings → Developer settings → Personal access tokens).
- **C&C stripped** — `c2_server/`, `c2_ui/`, `paper_trader.py`, `live_game.py`, `README_C2.md` removed from this branch (`git rm -r`).
- **PII** — commits are authored `Cluster0 <fourmou.m@gmail.com>` (`git log --all --format="%an <%ae>"`). For a fully anonymous public history, rewrite authors with `git filter-repo` / `filter-branch` before pushing, or publish via a fresh repo initialised from this branch’s tree.
- **Ignored artefacts** — `run/`, `data/`, `*.p`, `*.h5`, `*.png`, `venv/` remain ignored.

To verify locally: `git grep -n "ghp_\|BINANCE_API"`, `git log --all --full-history -- ".env"`, `git remote -v`.

---

## Citation & Related Work

If this line of work is useful, please cite the paper that formalised the problem we observed:

> **Exploitation Problem in World Models** — https://arxiv.org/pdf/2605.15960

And this repository (research track):

```
@software{azt_research_2026,
  title  = {AlphaZero Trader — Research Track: World Model Exploitation},
  author = {Cluster0},
  year   = {2026},
  url    = {https://github.com/clusterO/alphazerotrader},
  note   = {Branch research/track-2-world-model-fix, commit 500ade2 + roadmap}
}
```

---

## License

GPL-3.0 — see `LICENSE`. The Transformer/MCTS/world-model code is free to reuse; the private C&C deploy layer is not included.

---

## Disclaimer

Not financial advice. This is a research artefact demonstrating a known failure mode of model-based planning in non-stationary markets. Backtests are not indicative of live performance. Do not trade real capital on these models without independent risk management.

---

## Contact / Contributing

Open an issue or PR on this branch with an experiment log entry. For the private deploy track (`deploy/track-1-long-flat`), contact the maintainers separately — it is not part of this publication.
