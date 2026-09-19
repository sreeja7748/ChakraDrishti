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
The goal is to learn a scanning policy that achieves more successful interceptions than a simple fixed scanning strategy.

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

 ---

## ✨ Key Features
- Adaptive frequency-band selection
- Tabular Q-learning scheduler
- Limited-sensing simulation
- Multiple simulated RF emitter behaviors
- Periodic transmission pattern detection
- Round Robin baseline
- Random scheduler
- Epsilon-Greedy scheduler
- UCB scheduler
- PeriodicLock scheduler
- Performance benchmarking
- Probability of Detection (Pd)
- Probability of False Alarm (Pfa)
- Interception Ratio
- Average Interception Rate
- Average Interception Time
- Prediction accuracy metrics
- Standalone dashboard for result visualization

---

## 🧠 How ChakraDrishti Works

ChakraDrishti consists of four main components:

```text
┌───────────────────────┐
│       Emitters        │
│                       │
│ Fixed Frequency       │
│ Frequency Agile       │
│ Periodic Scan         │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│    RF Environment     │
│                       │
│ Creates hidden RF     │
│ ground-truth state    │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│       Receiver        │
│                       │
│ Scans ONE band        │
│ and returns HIT/MISS  │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│      Scheduler        │
│                       │
│ Q-Learning /          │
│ Round Robin / UCB /   │
│ Other strategies      │
└───────────┬───────────┘
            ↓
        Next Scan
```
The scheduler does not receive the complete RF ground truth.

It only receives the result of the frequency band that it actually scanned.

---

## 📡 RF Emitter Simulation

ChakraDrishti simulates multiple types of emitters.

### 1. Fixed Frequency Emitter

A fixed-frequency emitter operates on a specific frequency band and can transmit in bursts.

```text
Band: 3

Time →   0 1 2 3 4 5 6 7
         · · █ █ █ · · ·
```
Its transmission behavior is controlled by parameters such as:

- Frequency band
- Burst-start probability
- Burst length
  
### 2. Frequency Agile Emitter

A frequency-agile emitter can transmit on different frequency bands.

```text
Time →   0 1 2 3 4 5 6 7
Band →   2 · 6 1 · 7 3 ·
```
This represents an emitter whose operating frequency changes over time.

### 3. Periodic Scan Emitter

The periodic emitter follows a predefined repeating scanning pattern.

For example:

```text
B0 → B0 → B3 → B3 → B5 → B5 → B7 → B7
          ↑
       repeats
```
This allows ChakraDrishti to evaluate whether a scheduler can identify and exploit temporal regularities.

---

## 📥 Receiver Model

The receiver has limited sensing capability.

At each timestep, it scans only one frequency band.

The receiver produces:
```text
HIT
```
when an active transmission is detected, or:
```text
MISS
```
when no transmission is detected.

Detection is probabilistic and controlled using:

```python
p_detect
p_false_alarm
```
For example:
```text
p_detect = 0.90
p_false_alarm = 0.05
```
means the receiver has a 90% probability of detecting a transmission on a scanned active band and a 5% probability of producing a false alarm on an inactive band.

---

### 🤖 Q-Learning

ChakraDrishti uses tabular Q-learning as one of its adaptive scheduling strategies.

The learning structure can be represented as:

```text
             Q-Learning
                  │
                  ↓
        ┌──────────────────┐
        │      State       │
        │                  │
        │ Current phase    │
        └────────┬─────────┘
                 ↓
        ┌──────────────────┐
        │      Action      │
        │                  │
        │ Select RF band   │
        └────────┬─────────┘
                 ↓
        ┌──────────────────┐
        │    Observation   │
        │                  │
        │     HIT/MISS     │
        └────────┬─────────┘
                 ↓
        ┌──────────────────┐
        │      Reward      │
        │                  │
        │ HIT → 1          │
        │ MISS → 0         │
        └────────┬─────────┘
                 ↓
        Update Q-Table
```
The Q-learning update follows:
```text
Q(s,a) ← Q(s,a) +
         α [r + γ max Q(s',a') − Q(s,a)]
```
where: 
| Parameter | Meaning                 |
| --------- | ----------------------- |
| `s`       | Current state           |
| `a`       | Selected frequency band |
| `r`       | Reward                  |
| `s'`      | Next state              |
| `α`       | Learning rate           |
| `γ`       | Discount factor         |

In the current implementation, the state is represented by the phase of the observation cycle:
```python
state = t % period
```
The action corresponds to the selected frequency band.
