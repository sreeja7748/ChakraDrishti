# ChakraDrishti

### Adaptive RF/EW Spectrum Scanning using Reinforcement Learning

ChakraDrishti is an adaptive RF/EW spectrum scanning system designed to intelligently select which frequency band to monitor at each timestep under limited sensing capacity.

The system simulates different RF emitter behaviors and uses **tabular Q-learning** to learn which frequency bands are more valuable to scan during different phases of an observation cycle. It also includes temporal pattern detection through a **PeriodicLock scheduler** for identifying recurring transmission patterns.

---
