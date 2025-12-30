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

```
pip install --upgrade pip
pip install -r requirements.txt
```
The code is compatible with Python 3.9+ and supports CPU and CUDA GPU execution.

The repository contains three experimental settings, corresponding directly to the thesis chapters.

### Experiment 1: Attacker-Only Honeypot Detection
#### Train attacker PPO
```
python HoneypotAttacker/train_ppo.py --total-timesteps 200000
```

#### Evaluate attacker performance
```
python HoneypotAttacker/eval_ppo.py --model-path models/attacker_ppo_latest.pt --episodes 50
```

### Experiment 2: Defender-Only Honeypot Concealment
#### Train defender PPO
```
python HoneypotDefender/train_ppo.py --total-timesteps 1000000
```

#### Evaluate defender (PPO or random baseline)
```
python HoneypotDefender/eval_ppo.py --model-path models/defender_ppo_latest.pt --episodes 20
```
Or evaluate a random baseline:
```
python HoneypotDefender/eval_ppo.py --random
```

### Experiment 3: Multi-Agent Attacker–Defender Arena
#### Train multi-agent PPO
```
python HoneypotArena/train_multi_agent_ppo.py --total-timesteps 500000
```

#### Evaluate trained agents
```
python HoneypotArena/eval_multi_agent.py --attacker-model models_multi/att_temp.pt --defender-model models_multi/def_temp.pt --episodes 50
```

### Contact
Valentijn Ouwehand <br>
Leiden University <br>
Bachelor Computer Science <br>
v.n.ouwehand@umail.leidenuniv.nl
