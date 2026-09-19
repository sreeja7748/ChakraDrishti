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
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]
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

---

## 🔄 PeriodicLock

ChakraDrishti also contains a PeriodicLock scheduler.

Unlike Q-learning, PeriodicLock attempts to identify recurring temporal patterns from observed hits.

For example:

```text
Observed hits:

2 ───── 10 ───── 18 ───── 26

Gaps:

8       8        8

Estimated period ≈ 8
```
The scheduler can then use the discovered timing pattern to focus scanning around predicted activity.

This provides a different approach to adaptive scheduling:
```text
Q-Learning
"What action gives me better reward?"

PeriodicLock
"Does the observed activity repeat periodically?"
```

---

## 🧪 Scheduling Strategies

ChakraDrishti includes multiple scheduling approaches for comparison.

### Round Robin

Scans frequency bands sequentially:
```
0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
```
### Random
Selects frequency bands randomly.

### Epsilon-Greedy
Balances exploration and exploitation.
```text
Explore → Try different bands
Exploit → Select a high-value band
```

### UCB
Uses an Upper Confidence Bound strategy to balance exploration and exploitation.

### Q-Learning
Learns a frequency-selection policy based on observed rewards.

### PeriodicLock
Attempts to exploit recurring temporal patterns.

---

## 📊 Performance Metrics

The system evaluates scheduler performance using several metrics.

### Probability of Detection — Pd
Measures how often actual transmissions are successfully detected.

```
Pd = TP / (TP + FN)
```
Where:

- TP = True Positives
- FN = False Negatives

### Probability of False Alarm — Pfa
Measures how often the receiver reports a transmission when none exists.

```
Pfa = FP / (FP + TN)
```
Where:

- FP = False Positives
- TN = True Negatives

### Interception Ratio
Measures the fraction of all actual transmission events that were successfully intercepted.

```
Interception Ratio = True Positive Interceptions / Total Transmission Events
```
This differs from Pd because the receiver is only scanning one band at a time.

### Average Interception Rate
Measures successful interceptions relative to the number of simulation timesteps.

### Average Interception Time
Measures the average time between successful interceptions.

---

## 🧩 Project Architecture

```text
                    ┌─────────────────────┐
                    │      Emitters       │
                    │                     │
                    │ Fixed Frequency     │
                    │ Frequency Agile     │
                    │ Periodic Scan       │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │  RF Environment     │
                    │                     │
                    │ Hidden Truth State  │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │      Receiver       │
                    │                     │
                    │ One-band sensing    │
                    └──────────┬──────────┘
                               │
                         HIT / MISS
                               │
                               ↓
                    ┌─────────────────────┐
                    │     Scheduler       │
                    │                     │
                    │ Q-Learning          │
                    │ PeriodicLock        │
                    │ UCB                 │
                    │ Epsilon-Greedy      │
                    │ Random              │
                    │ Round Robin         │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │    Performance      │
                    │     Metrics         │
                    └─────────────────────┘
```

---

## 📁 Project Structure

```text
Electronic_Warfare_Scanner/
│
├── demo_stage1.py
├── demo_stage2.py
├── demo_stage3.py
├── demo_stage4.py
├── demo_stage5.py
├── demo_stage6.py
├── demo_stage7.py
│
├── emitters.py
├── environment.py
├── receiver.py
├── schedulers.py
├── simulate.py
├── train_qlearning.py
├── metrics.py
│
├── inspect_h5.py
└── index.html
```
### File Description
| File                                | Purpose                                           |
| ----------------------------------- | ------------------------------------------------- |
| `emitters.py`                       | Defines simulated RF emitter behaviors            |
| `environment.py`                    | Creates the RF environment and ground-truth state |
| `receiver.py`                       | Simulates receiver sensing                        |
| `schedulers.py`                     | Contains different scheduling strategies          |
| `train_qlearning.py`                | Trains the Q-learning scheduler                   |
| `simulate.py`                       | Runs simulation experiments                       |
| `metrics.py`                        | Calculates performance metrics                    |
| `demo_stage1.py` – `demo_stage7.py` | Demonstration and evaluation stages               |
| `index.html`                        | Standalone visualization dashboard                |
| `inspect_h5.py`                     | Utility script for inspecting H5 files            |

---

## ⚙️ Configuration
The current benchmark configuration includes:

```python
NUM_BANDS = 8
NUM_STEPS = 60

P_DETECT = 0.9
P_FALSE_ALARM = 0.05

NUM_TEST_SEEDS = 30
```
The Q-learning training configuration includes parameters such as:
```python
training_epsilon = 0.15
alpha = 0.1
gamma = 0.5
```
The trained scheduler switches to exploitation during evaluation.

---

## 📸 Project SnapShot
<img width="1899" height="999" alt="Screenshot (592)" src="https://github.com/user-attachments/assets/c296e741-e967-400a-b273-c1157490c039" />
<img width="1907" height="1019" alt="Screenshot (593)" src="https://github.com/user-attachments/assets/aaed7da2-cb9c-4dce-824d-c94a2d7bd2af" />
<img width="1890" height="1018" alt="Screenshot (595)" src="https://github.com/user-attachments/assets/865e5ab5-e3a9-4b95-9972-d1e0a4844fe5" />
<img width="1896" height="1029" alt="Screenshot (596)" src="https://github.com/user-attachments/assets/689adf9e-5938-4d30-b3f8-50ff1e9465a8" />
<img width="1907" height="1029" alt="Screenshot (597)" src="https://github.com/user-attachments/assets/55df89c2-add2-493a-b006-df857d7e36f1" />
<img width="1901" height="1004" alt="Screenshot (598)" src="https://github.com/user-attachments/assets/8b819189-e0f3-4efd-8f63-780e7b042ed6" />
<img width="1900" height="1007" alt="Screenshot (599)" src="https://github.com/user-attachments/assets/a74ac125-5e69-4d51-989d-781f915d683b" />

---

## 🚀 Running the Project
### 1. Clone the repository

```bash
git clone https://github.com/sreeja7748/ChakraDrishti.git
cd Electronic_Warfare_Scanner
```
### 2. Run the demonstrations
For example:
```bash
python demo_stage6.py
```
Individual stages can also be executed:
```bash
python demo_stage1.py
python demo_stage2.py
python demo_stage3.py
python demo_stage4.py
python demo_stage5.py
python demo_stage6.py
python demo_stage7.py
```
### 3. Open the dashboard
Open:
```
index.html
```
in a web browser to view the standalone visualization.

---

## 🧪 Experimental Setup

The evaluation uses multiple independent test scenarios.

The current Stage 7 benchmark uses:
```
60 simulation ticks
8 frequency bands
30 held-out test scenarios
90% detection probability
5% false-alarm probability
500 Q-learning training episodes
```

The Q-learning scheduler is compared against multiple baseline and alternative scheduling strategies.

---

## 🔐 Security Considerations

ChakraDrishti is currently designed as a local simulation prototype rather than a network-exposed production service.

The core simulator primarily uses Python's standard library, minimizing third-party dependency exposure.

Security considerations for future deployment include:

- Pinning external dependencies
- Auditing dependencies for known vulnerabilities
- Validating all external inputs
- Restricting API access
- Limiting computational resources
- Keeping hidden ground truth isolated from external interfaces
- Protecting the learned Q-table
- Avoiding unnecessary network exposure
- Running the system locally where possible
- Removing development artifacts before deployment

Python's *random* module is used for simulation behavior and reproducibility. It is not used for cryptographic security.

---

## ⚠️ Important Scope

ChakraDrishti is currently a simulation and scheduling prototype.

It does not currently perform:

- Real RF signal acquisition
- Hardware-based SDR reception
- Waveform classification
- Automatic emitter identification
- Real-world spectrum interception
- RF transmission or jamming

The simulated environment provides controlled ground truth for evaluating the receiver and scheduling algorithms.

---

## 💡 Innovation

The primary focus of ChakraDrishti is not inventing a new reinforcement-learning algorithm.

Instead, the project explores the application of lightweight reinforcement learning to an RF/EW scanning problem where:

- The receiver has limited sensing capacity.
- Only one frequency band can be scanned at a time.
- The scheduler must make sequential decisions.
- Feedback from previous scans can influence future decisions.
- Temporal patterns can be exploited to improve scanning efficiency.

The project also compares reinforcement learning with several traditional and alternative scheduling strategies.

---

## 🎯 Future Scope

Possible future extensions include:

### Hardware Integration

Integration with Software Defined Radio (SDR) hardware for real RF spectrum observations.

### Real-Time Spectrum Monitoring

Replace simulated emitter states with real receiver measurements.

### Signal Classification

Extend the system from simple signal presence detection to:
```text
Signal Detection
      ↓
Feature Extraction
      ↓
Signal Classification
      ↓
Emitter Identification
```
### Advanced RL

Explore:

- Deep Q-Networks
- Contextual bandits
- Multi-agent reinforcement learning
- Partially Observable Markov Decision Processes
- Adaptive Threat Prioritization

Assign different priorities to detected signals based on their behavior and observed characteristics.

### Hardware-in-the-Loop Testing

Combine simulated environments with real SDR hardware for controlled experiments.

---

## 🏆 Results

The current Stage 7 benchmark evaluates the schedulers over 30 held-out scenarios.

The project dashboard provides comparative results for:

```text
Q-Learning
Round Robin
Random
Epsilon-Greedy
UCB
PeriodicLock
```
The exact numerical results are generated by the experiment configuration and should be regenerated when the simulation parameters or implementation change.

---

## 🛠️ Technology Stack
### Programming
- Python
- HTML
- CSS
- JavaScript
### Machine Learning
- Reinforcement Learning
- Tabular Q-learning
### Simulation
- Custom RF environment
- Simulated RF emitters
- Probabilistic receiver model
### Visualization
- Standalone HTML dashboard
- JavaScript
- SVG-based visualization

---

## 📚 Concepts Used
- Radio Frequency (RF)
- Electronic Warfare (EW)
- Spectrum Monitoring
- Frequency Scanning
- Reinforcement Learning
- Q-learning
- Exploration vs Exploitation
- Temporal Pattern Detection
- Probability of Detection
- Probability of False Alarm
- Signal Interception
- Sequential Decision Making
- Limited Sensing

---

## 📜 Disclaimer

ChakraDrishti is an educational and research-oriented simulation project intended for controlled experimentation with RF/EW scheduling concepts.

It does not implement real-world RF transmission, jamming, or unauthorized spectrum interception.
