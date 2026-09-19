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

## 𖦏 What's actually in this repo
 
| Layer | File(s) | Role |
|---|---|---|
| **Environment** | `emitters.py`, `environment.py` | Simulated fixed-frequency, frequency-agile, and periodic-scan emitters generate ground truth across all bands, every tick |
| **Receiver** | `receiver.py` | Physically limited to one band per tick; models a realistic imperfect sensor (configurable detection / false-alarm rates) |
| **Schedulers** | `schedulers.py` | Six interchangeable strategies (see below) — every one decides using *only* its own scan history, never the environment's ground truth |
| **Training** | `train_qlearning.py` | Offline Q-learning training loop — many practice episodes, epsilon-greedy exploration, TD updates |
| **Metrics** | `metrics.py` | Pd, Pfa, interception ratio, avg intercept rate/time, computed independently *after* a run — the scheduler never sees this |
| **Evaluation runner** | `simulate.py`, `demo_stage1.py`–`demo_stage7.py` | Reusable simulation harness + one script per development stage, building up from a naive baseline to the final comparison |
| **Interactive dashboard** | `index.html` | Self-contained live demo — animated radar scope + spectrum grid, judge-facing, zero build step |
 
