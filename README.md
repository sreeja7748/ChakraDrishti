# 📡 ChakraDrishti

### Adaptive RF/EW Spectrum Scanning using Reinforcement Learning

ChakraDrishti is an adaptive RF/EW spectrum scanning system designed to intelligently select which frequency band to monitor at each timestep under limited sensing capacity.

The system simulates different RF emitter behaviors and uses **tabular Q-learning** to learn which frequency bands are more valuable to scan during different phases of an observation cycle. It also includes temporal pattern detection through a **PeriodicLock scheduler** for identifying recurring transmission patterns.

---

## 📌 Overview

In a real RF/Electronic Warfare environment, a receiver may not be able to monitor every frequency band simultaneously.

This creates a decision problem:

> **Which frequency band should be scanned next to maximize the probability of detecting an active transmission?**

ChakraDrishti models this problem as a sequential decision-making task.

At every timestep:

```text
        ┌─────────────────────┐
        │ Select Frequency    │
        │      Band           │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │ Scan Selected Band  │
        └──────────┬──────────┘
                   ↓
             HIT / MISS
                   ↓
        ┌─────────────────────┐
        │ Calculate Reward    │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │ Update Q-Table      │
        └──────────┬──────────┘
                   │
                   └──────→ Next Decision
```

---

