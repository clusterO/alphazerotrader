# Agent Suggestions for Future Phases

This file tracks technical improvements identified by the builder agent during Phase 2/3 that fall outside the immediate scope of the consultant's gated plan.

## 1. Feature Engineering
- **Normal Distribution**: Z-score normalization per window is good, but for some features like `vol_delta`, log-scaling or robust scaling might be better to handle outliers.
- **Time Features**: Add sin/cos encoding for "hour of day" and "day of week" to the state tensor.

## 2. World Model Details
- When implementing the World Model, consider a VAE (Variational Autoencoder) or a simple GMM (Gaussian Mixture Model) to simulate price returns and volatility, rather than just a linear drift.

## 3. Training Efficiency
- **Gradient Clipping**: Essential for Transformers to prevent explosion on market shocks.
- **Learning Rate Scheduler**: Decay the learning rate as iterations increase.

## 4. Multi-Symbol Training
- Once v3.1 is stable, train on a basket of symbols (ETH, SOL, etc.) simultaneously to build a more robust, generalized feature extractor.
