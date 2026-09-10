## TODO (Research Track — `research/track-2-world-model-fix`)

> This branch is the **public research track**. The private `deploy/track-1-long-flat` C&C layer is intentionally excluded here.

- [ ] **Exploitation defenses** — see `README.md` Roadmap (Categories A/B/C); first fix: rollout return ceiling
- Mechanistic Interpretability

## IDEAS

#### Feature Engineering — 10 New Features
Add these in one batch. Keep the existing tensor structure but expand Channel 1:
# Time encoding (cyclical, so the model understands hour 23 is close to hour 0)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
day_sin = np.sin(2 * np.pi * day_of_week / 7)
day_cos = np.cos(2 * np.pi * day_of_week / 7)

**Volatility regime**
atr_14 = average_true_range(high, low, close, period=14)
bb_width = (bb_upper - bb_lower) / bb_mid # Bollinger Band width

**Volume features**
volume_ratio = volume / volume.rolling(20).mean() # relative volume
vwap_dev = (close - vwap) / vwap # deviation from VWAP

**Momentum**
log_return = np.log(close / close.shift(1))
roc_10 = (close - close.shift(10)) / close.shift(10)
Normalize each feature with rolling z-score (mean and std over the window) rather than global normalization — this handles non-stationarity better.

**Channel 2**
Another upper timeframe

#### Continuous Action Space — Defer with a Gate
Don't implement now. Implement a gate: once your discrete agent achieves annualized Sharpe > 1.0 consistently on held-out data for 4+ consecutive evaluation rounds, 
that's the trigger to begin the continuous action space branch. Build it in parallel, don't replace the discrete agent 
— use the discrete agent as the baseline to beat.
The continuous action implementation when the time comes:
# Action head outputs: [position_size] in [-1, 1]
# -1.0 = maximum short
# -0.3 = small short  
# 0.0 = flat
# +0.5 = medium long
# +1.0 = maximum long

# Reward modification needed:
# transaction_cost scales with |new_position - old_position|
# not just on direction change
Synthetic Data Ablation — One-Time Validation
Before the next architecture change (if any), run this once. Create two synthetic environments:
Environment 1 — Mean reverting (OU process): known edge is mean reversion, so a correctly functioning agent should learn to buy dips and sell rallies.
Environment 2 — Trending (GBM with drift): known edge is momentum, agent should learn to follow trend.
If your current architecture can learn the correct policy on both synthetic environments, 
the architecture is sound and any remaining real-data failure is a features or data problem. 
If it can't learn on synthetic data where the edge is guaranteed and known, you have a learning mechanism problem that more features won't fix.

Cons
Serious challenges:
Non-stationarity — market dynamics shift over time (regimes change). A strategy that worked in 2018 may fail in 2022. AlphaZero doesn't face this; 
chess rules never change.
Partial observability — you never see the full "board." Other participants' intentions, order flow, institutional positions are hidden.
Sparse and noisy rewards — in chess, reward is clear at game end. In trading, daily P&L is noisy and slow to reveal true edge.
Overfitting to history — the agent can memorize past market patterns that won't repeat (backtest overfitting is the #1 killer of trading systems).
Market impact — your own trades move prices, especially at scale. The environment reacts to you, breaking the stationarity assumption.
World model error compounds — small errors in predicting next market states explode over multi-step rollouts, making deep MCTS planning unreliable.
Sample efficiency — RL needs enormous amounts of experience. Markets generate limited data compared to self-play games.
Transaction costs — a theoretically profitable strategy can be destroyed by slippage and commissions at high frequency.

Promising Design Ideas
Idea 1 — Regime-Aware Agent
Train separate sub-policies for different market regimes (trending, mean-reverting, high-volatility). 
A meta-controller (also learned) selects which sub-policy to deploy based on detected regime. This addresses non-stationarity directly.
Idea 2 — Opponent Modeling
Treat institutional order flow, market makers, and retail sentiment as distinct "opponents." Model their likely behavior and plan accordingly 
— closer to true game-theoretic thinking.
Idea 3 — Hierarchical RL
Two levels of decision-making:
High-level policy — decides strategy and risk allocation over days/weeks
Low-level policy — handles trade execution over seconds/minutes
This mirrors how a human portfolio manager (high level) works with a trader (low level).
Idea 4 — Uncertainty-Aware Planning
Augment the value head to output a distribution over future returns, not just a point estimate. 
The agent then avoids actions with high variance even if the mean looks good — naturally risk-aware behavior.
Idea 5 — Causal World Model
Instead of a purely statistical world model, build one that encodes causal relationships (earnings → volatility, Fed rate → bonds → equities). 
This generalizes better to unseen regimes because it understands why things happen, not just what follows what.
Idea 6 — Continual Learning
Never stop training. As new market data arrives, continuously fine-tune the network 
— but with mechanisms to prevent forgetting old knowledge (elastic weight consolidation, experience replay with old data). 
The agent stays current without catastrophic forgetting.

