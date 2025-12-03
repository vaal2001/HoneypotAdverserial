from __future__ import annotations
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from HoneypotDefender.env.env import HoneypotDefenderEnv
from HoneypotDefender.rl.ppo_agent import PPOAgent
from HoneypotDefender.utils.logger import SimpleLogger


class RolloutBuffer:
    def __init__(self, num_steps: int, obs_dim: int):
        self.num_steps = num_steps
        self.obs = np.zeros((num_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(num_steps, dtype=np.int64)
        self.logprobs = np.zeros(num_steps, dtype=np.float32)
        self.rewards = np.zeros(num_steps, dtype=np.float32)
        self.dones = np.zeros(num_steps, dtype=np.float32)
        self.values = np.zeros(num_steps, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/defender_ppo_latest.pt",
    )
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    logger = SimpleLogger()

    # -------------------------
    # ENV INIT
    # -------------------------
    env = HoneypotDefenderEnv()
    obs, info = env.reset()
    obs_flat = env.flatten_obs(obs)

    obs_dim = obs_flat.shape[0]
    n_actions = env.action_space.n
    N_max = env.N_max
    F_features = env.F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # AGENT INIT (GNN-based PPO)
    # -------------------------
    agent = PPOAgent(obs_dim, n_actions, N_max, F_features).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    buffer = RolloutBuffer(args.num_steps, obs_dim)

    global_step = 0
    episode_returns = []

    start_time = time.time()

    # -------------------------
    # TRAINING LOOP
    # -------------------------
    while global_step < args.total_timesteps:
        # ROLLOUT
        for t in range(args.num_steps):
            buffer.obs[t] = obs_flat

            obs_tensor = torch.tensor(
                obs_flat, dtype=torch.float32, device=device
            )

            with torch.no_grad():
                a_t, logp_t, ent_t, v_t = agent.get_action_and_value(obs_tensor)

            action = int(a_t.item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.actions[t] = action
            buffer.logprobs[t] = float(logp_t.cpu().item())
            buffer.rewards[t] = float(reward)
            buffer.dones[t] = float(done)
            buffer.values[t] = float(v_t.cpu().item())

            global_step += 1
            obs = next_obs
            obs_flat = env.flatten_obs(next_obs)

            if done:
                episode_returns.append(sum(buffer.rewards[: t + 1]))
                obs, info = env.reset()
                obs_flat = env.flatten_obs(obs)

            if global_step >= args.total_timesteps:
                break

        # BOOTSTRAP
        obs_tensor = torch.tensor(
            obs_flat, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            _, next_value = agent(obs_tensor)

        next_value = float(next_value.cpu().item())

        advantages = np.zeros_like(buffer.rewards)
        last_gae = 0.0

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                next_nonterminal = 1.0 - buffer.dones[t]
                next_values = next_value
            else:
                next_nonterminal = 1.0 - buffer.dones[t + 1]
                next_values = buffer.values[t + 1]

            delta = (
                buffer.rewards[t]
                + args.gamma * next_values * next_nonterminal
                - buffer.values[t]
            )
            advantages[t] = last_gae = (
                delta
                + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            )

        returns = advantages + buffer.values

        # TORCH BATCHES
        b_obs = torch.tensor(buffer.obs, dtype=torch.float32, device=device)
        b_actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
        b_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32, device=device)
        b_returns = torch.tensor(returns, dtype=torch.float32, device=device)
        b_adv = torch.tensor(advantages, dtype=torch.float32, device=device)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        batch_size = args.num_steps
        minibatch = batch_size // 4
        idxs = np.arange(batch_size)

        # PPO UPDATE
        for epoch in range(args.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch):
                end = start + minibatch
                mb = idxs[start:end]

                logits, values = agent(b_obs[mb])
                dist = Categorical(logits=logits)

                new_logprob = dist.log_prob(b_actions[mb])
                entropy = dist.entropy().mean()

                ratio = (new_logprob - b_logprobs[mb]).exp()
                s1 = ratio * b_adv[mb]
                s2 = torch.clamp(
                    ratio, 1 - args.clip_eps, 1 + args.clip_eps
                ) * b_adv[mb]
                policy_loss = -torch.min(s1, s2).mean()

                value_loss = ((values - b_returns[mb]) ** 2).mean()

                loss = (
                    policy_loss
                    + args.vf_coef * value_loss
                    - args.ent_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        avg_return = float(b_returns.mean().item())
        logger.log(
            step=global_step,
            avg_return=f"{avg_return:.3f}",
            episodes=len(episode_returns),
        )

        # SAVE CHECKPOINT
        torch.save(
            {
                "model_state_dict": agent.state_dict(),
                "obs_dim": obs_dim,
                "n_actions": n_actions,
                "N_max": N_max,
                "F_features": F_features,
            },
            args.save_path,
        )

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
