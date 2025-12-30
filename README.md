# Adversarial Honeypot Detection and Deployment with Reinforcement Learning
## Source code for the Bachelor Thesis

### Author
Valentijn Ouwehand <br>
Bachelor Computer Science, University of Leiden, 2026

#### Supervisors
Dr. Ir. E. Makri <br>
Dr. T.M. Moerland

---

### Overview
This repository contains the source code corresponding to the thesis titled **"Adversarial Honeypot Detection and Deployment with Reinforcement Learning"**

The purpose of this code is to support the implementation, experiments, and results described in the thesis and to enable reproducibility of the presented work.

---

### Abstract / Project Summary
This thesis explores the application of Reinforcement Learning (RL) in the context of honeypot detection and deployment in a simulated network. Honeypots are deceptive systems designed to attract and engage attackers. However, the increasing use of sophisticated detection techniques presents new challenges. The goal of this thesis is to investigate the interaction between a Reinforcement Learning attacker and defender in a honeypot network. The attacker attempts to learn an optimal policy for detecting honeypots in a network, and the defender aims to find the optimal strategy for disguising these honeypots. Through a series of experiments in a simulated network environment, this research evaluates the interaction between these two agents. The findings suggest that RL-based strategies can enhance honeypot detection techniques, even when faced with an opposing defender agent.

---

### Repository Overview
This repository is composed of three parts "HoneypotAttacker", "HoneypotDefender", "HoneypotArena" corresponding to the three experiments described and executed in the report. In order to reproduce the experiments it is important to have the correct dependencies. For this use a virtual environment.

```
python3 -m venv venv
source venv/bin/activate
```

next explain how to call the different dexperiments and evaluations
