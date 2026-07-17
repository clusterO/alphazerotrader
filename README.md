# AlphaZero Trader — Track 2: Research (World Model Fix)

> **Branch:** `research/track-2-world-model-fix`
> **Status:** 🔬 Research — World model preserved, exploitation defense in progress

---

## Project Split: Two Tracks, Two Futures

| Branch | Purpose | World Model | Status |
|---|---|---|---|
| **`deploy/track-1-long-flat`** | Production deployable system | ❌ Removed | 🟢 Active |
| **`research/track-2-world-model-fix`** ← *you are here* | Architecture research & exploitation fixes | ✅ Active | 🔬 Research |

Switch to the deploy branch:
```bash
git checkout deploy/track-1-long-flat
```

---

## Context: Why This Branch Exists

The World Model (Conv1D+LSTM transition predictor) was introduced to allow MCTS to plan beyond historical data. Architecturally it was sound — the gates all passed, the value head separated correctly, and MCTS divergence appeared as expected.

The failure was **exploitation**. MCTS with 600 simulations optimizing through an imperfect world model found hallucinated action sequences producing 30–50% episode returns. The policy fully absorbed those patterns.

Fair Mode backtest confirmed: v18 and v19 fail on all four regimes. The agent learned to play the world model's fantasy market, not the real one.

This branch preserves the full world model architecture and contains the roadmap for fixing the exploitation problem. It is a **research sandbox** — do not retrain production models here.

---

## Exploitation Problem: Root Cause Analysis

### Why MCTS Exploits a World Model
```
Real market trajectory:     price moves ±0.3% per tick on average
WM prediction error:        ±0.8% residual per tick (compounding)
MCTS 20-step rollout error: potentially ±16% accumulated drift

→ MCTS finds "fantasy paths" where WM error consistently favors Long
→ Policy gradient absorbs those paths as real signal
→ Model trained on fantasy converges to garbage on real data
```

### Three Categories of Fixes

#### Category A — Limit MCTS Exploitation of WM Errors
These are surgical: they don't change the architecture, just constrain how MCTS uses the WM.

- [ ] **Rollout Return Ceiling**: Cap the maximum return any WM rollout can report to `+/- 2 * historical_sigma`. Any path exceeding this is fantasy; clip it. *(Estimated impact: HIGH)*
- [ ] **WM Depth Limit**: Reduce WM rollout depth from 5 to 2. Errors compound geometrically. *(Estimated impact: MEDIUM)*
- [ ] **WM Confidence Score**: Add a per-step prediction uncertainty estimate (MC Dropout or ensemble variance). Discount rollout value by accumulated uncertainty. *(Estimated impact: HIGH, complex)*

#### Category B — Training Signal Correction
These address how the policy learns from MCTS outputs on WM data.

- [ ] **KL Divergence Regularization**: Add a KL term between the policy's distribution on WM-rollout states and historical-data states. Forces WM-trained policy to stay close to what real data teaches. *(Estimated impact: HIGH)*
- [ ] **Value Head Reality Anchoring**: During training, blend WM-rollout values (from MCTS) with direct NN values on real historical next states. Prevents value head from calibrating purely on fantasy returns. *(Estimated impact: HIGH)*
- [ ] **WM-only vs History-only Memory Split**: Keep two replay buffers — one for WM-augmented episodes, one for pure historical. Train with a 25/75 blend. *(Estimated impact: MEDIUM)*

#### Category C — World Model Architecture Improvements
These make the WM more accurate and less exploitable.

- [ ] **Residual WM (predict delta, not full state)**: Predict `Δstate` not next_state. Errors in delta space compound slower. *(Estimated impact: MEDIUM)*
- [ ] **Ensemble WM (3 models, take pessimistic value)**: Use 3 WM models trained on different data splits. Use the *minimum* predicted value across ensemble for MCTS expansion. *(Estimated impact: HIGH, expensive)*
- [ ] **WM Error Feedback Loop**: After each real episode, compute WM prediction error per step and use it to down-weight WM rollout confidence. Continuously calibrating uncertainty. *(Estimated impact: HIGH)*

---

## Recommended First Experiment: Rollout Return Ceiling

The lowest-risk, highest-expected-impact fix. Implement in `agent.py` during WM expansion:

```python
# In the batched WM backfill loop (agent.py):
WM_RETURN_CEILING = 2.0 * historical_sigma  # e.g. 0.06 for 1h BTC
v = np.clip(v, -WM_RETURN_CEILING, WM_RETURN_CEILING)
```

**Expected result**: MCTS can no longer find 30–50% fantasy paths. The maximum reward it can assign to any WM rollout is bounded to realistic historical volatility. Policy should stop absorbing fantasy patterns.

**Gate**: Run Fair Mode backtest on bear_2022.csv. If drawdown is reduced and the agent doesn't immediately hit the 15% stop-out, Category A fixes are working.

---

## Experiment Log

| Date | Experiment | Hypothesis | Result |
|---|---|---|---|
| 2026-07-17 | Fair Mode backtest v18 & v19 | World model caused exploitation | ✅ CONFIRMED — total failure all regimes |
| — | Rollout ceiling (Category A) | Cap fantasy returns | — |
| — | KL regularization (Category B) | Anchor policy to real data | — |
| — | Ensemble WM (Category C) | Pessimistic rollouts | — |

---

## Files Changed Relative to Main

This branch is identical to `main` (the world model snapshot). No changes yet — experiments go here.

Key files for WM research:
| File | Role |
|---|---|
| `world_model.py` | Conv1D+LSTM transition predictor |
| `transition_memory.py` | Replay buffer for (s, a, s') transitions |
| `agent.py` | WM integration in batched MCTS expansion |
| `main.py` | WM retraining loop (every 2 iterations) |
| `game.py` | `takeActionWM` and `takeActionWM_FromPrediction` |
| `RESEARCH.md` | *(this document, for detailed notes)* |

---

## Running Experiments

```bash
# 1. Make your change (e.g. add rollout ceiling to agent.py)
# 2. Retrain for N iterations (start small, 5-10 iterations)
python3 main.py

# 3. Run Fair Mode backtest on all 4 regimes
python3 backtest.py --model run/models/BTC_USDT_1h_v0001.keras \
    --data data/backtest/bear_2022.csv

# 4. Log results in the Experiment Log above
```

---

## Contact

See `deploy/track-1-long-flat` branch for the production system.
