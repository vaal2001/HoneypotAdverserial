from env import HoneypotAdversarialEnv
import numpy as np

env = HoneypotAdversarialEnv()
obs, info = env.reset()

done = False
while not done:
    a_att = env.attacker_env.action_space.sample()
    a_def = env.defender_env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(
        {"attacker": a_att, "defender": a_def}
    )
    done = terminated or truncated

    print("r_att =", rew["attacker"], "r_def =", rew["defender"])
