import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from HoneypotAttacker.env.env import HoneypotDetectionEnv
from HoneypotAttacker.rl.ppo_agent import PPOAgent

class RolloutBuffer:
    def __init__(self, num_steps, obs_dim, mask_dim):
        self.num_steps = num_steps
        self.obs = np.zeros((num_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(num_steps, dtype=np.int64)
        self.logprobs = np.zeros(num_steps, dtype=np.float32)
        self.rewards = np.zeros(num_steps, dtype=np.float32)
        self.dones = np.zeros(num_steps, dtype=np.float32)
        self.values = np.zeros(num_steps, dtype=np.float32)
        self.masks = np.zeros((num_steps, mask_dim), dtype=np.bool_)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.92)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    env = HoneypotDetectionEnv()
    obs, _ = env.reset()
    mask = obs["action_mask"]

    obs_flat = env.flatten_obs(obs)
    obs_dim = obs_flat.shape[0]
    mask_dim = mask.shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPOAgent(obs_dim, n_actions, env.N_max, env.F).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    buffer = RolloutBuffer(args.num_steps, obs_dim, mask_dim)

    global_step = 0
    start_time = time.time()

    while global_step < args.total_timesteps:
        obs_flat = env.flatten_obs(obs)

        for t in range(args.num_steps):
            buffer.obs[t] = obs_flat
            buffer.masks[t] = mask

            obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                a_t, logp_t, _, v_t = agent.get_action_and_value(obs_tensor, action_mask=mask_tensor)

            action = a_t.item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_mask = next_obs["action_mask"]

            buffer.actions[t] = action
            buffer.logprobs[t] = logp_t.cpu().item()
            buffer.rewards[t] = reward
            buffer.dones[t] = float(done)
            buffer.values[t] = v_t.cpu().item()

            global_step += 1
            obs = next_obs
            obs_flat = env.flatten_obs(next_obs)
            mask = next_mask

            if done:
                obs, _ = env.reset()
                obs_flat = env.flatten_obs(obs)
                mask = obs["action_mask"]

            if global_step >= args.total_timesteps:
                break

        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, _, next_value = agent.get_action_and_value(obs_tensor, action_mask=mask_tensor)

        next_value = next_value.cpu().item()

        advantages = np.zeros_like(buffer.rewards)
        last_gae = 0.0

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                next_nonterminal = 1.0 - buffer.dones[t]
                next_values = next_value
            else:
                next_nonterminal = 1.0 - buffer.dones[t + 1]
                next_values = buffer.values[t + 1]

            delta = (buffer.rewards[t] + args.gamma * next_values * next_nonterminal - buffer.values[t])
            advantages[t] = last_gae = (delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae)

        returns = advantages + buffer.values

        b_obs = torch.tensor(buffer.obs, dtype=torch.float32, device=device)
        b_actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
        b_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32, device=device)
        b_masks = torch.tensor(buffer.masks, dtype=torch.bool, device=device)
        b_returns = torch.tensor(returns, dtype=torch.float32, device=device)
        b_adv = torch.tensor(advantages, dtype=torch.float32, device=device)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        batch_size = args.num_steps
        minibatch = batch_size // 4
        idxs = np.arange(batch_size)

        for _ in range(args.update_epochs):
            np.random.shuffle(idxs)

            for start in range(0, batch_size, minibatch):
                end = start + minibatch
                mb = idxs[start:end]

                logits, values = agent(b_obs[mb], action_mask=b_masks[mb])
                dist = Categorical(logits=logits)

                new_logprob = dist.log_prob(b_actions[mb])
                entropy = dist.entropy().mean()

                ratio = (new_logprob - b_logprobs[mb]).exp()
                s1 = ratio * b_adv[mb]
                s2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * b_adv[mb]
                policy_loss = -torch.min(s1, s2).mean()

                value_loss = ((values - b_returns[mb]) ** 2).mean()

                loss = (policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        print(f"[step={global_step}] avg_return={b_returns.mean().item():.3f}")

        torch.save({"model_state_dict": agent.state_dict(), "obs_dim": obs_dim, "n_actions": n_actions, "N_max": env.N_max, "F_features": env.F}, "models/attacker_ppo_latest.pt")

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.1f}s.")

if __name__ == "__main__":
    main()
