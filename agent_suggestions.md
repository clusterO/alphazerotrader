# AlphaZero Trader - Idea Tracking & Backlog

This file tracks implemented features and future architectural improvements identified during the research phases.

## 1. Current State (v3.2.9)
- **Model**: Transformer (64-dim, 4 heads, 2 layers).
- **Inference**: Batched MCTS (Batch size 16, 100 sims).
- **Environment**: High-speed NumPy engine.
- **Current Features**:
    - `log_return`
    - `rsi` (14)
    - `atr` (14)
    - `vol_10`, `vol_30` (Rolling Volatility)
    - `rel_high_low` (30-period relative position)
    - `vol_delta` (Volume change)
- **Portfolio Features**:
    - `position` (-1, 0, 1)
    - `pnl_since_entry`
    - `steps_held` (normalized)

## 2. Future Improvements (Phase 6+)

### Better Eyes (Perception)
- [ ] **Multi-Timeframe Context**: Add 4h or 1d candle data as secondary input channels to the Transformer.
- [ ] **Advanced Indicators**: MACD, Bollinger Band width, Ichimoku Cloud components.
- [ ] **Orderbook Depth**: If available, add bid/ask spread and depth imbalance.

### Risk & Reward Refinement
- [ ] **Drawdown Penalty**: Modify reward to penalize consecutive losing steps or max drawdown within an episode.
- [ ] **Volatility-Adjusted PnL**: Reward based on Sortino ratio logic rather than pure PnL.

### Training & Evaluation Efficiency
- [ ] **Evaluation Optimization**: Discuss finding the "Sweet Spot" between 5 and 30 evaluation episodes.
- [ ] **Asynchronous Collection**: Move self-play to a separate process from training.
- [ ] **Dynamic Sim Count**: Use fewer MCTS simulations early in training and increase them as the model matures.
- [ ] **Live Session Randomization**: Implement "Planned Session Length" in Paper/Live trading to mirror training randomization. This prevents the agent from staying in the same session indefinitely and forces it to re-evaluate portfolio state from a fresh "reset" periodically.

## 3. Active Research
- **Current Goal**: Validating generalization across Regimes (2025 Bull vs 2022 Bear).
