import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from HoneypotArena.multi_agent_env.multi_env import MultiAgentHoneypotEnv
from HoneypotAttacker.rl.ppo_agent import PPOAgent as AttackerPPOAgent
from HoneypotDefender.rl.ppo_agent import PPOAgent as DefenderPPOAgent

class AttackerRolloutBuffer:
    def __init__(self, num_steps, obs_dim, mask_dim):
        self.num_steps = num_steps
        self.obs = np.zeros((num_steps, obs_dim), dtype=np.float32)
        self.masks = np.zeros((num_steps, mask_dim), dtype=np.bool_)
        self.actions = np.zeros(num_steps, dtype=np.int64)
        self.logprobs = np.zeros(num_steps, dtype=np.float32)
        self.rewards = np.zeros(num_steps, dtype=np.float32)
        self.dones = np.zeros(num_steps, dtype=np.float32)
        self.values = np.zeros(num_steps, dtype=np.float32)

class DefenderRolloutBuffer:
    def __init__(self, num_steps, obs_dim):
        self.num_steps = num_steps
        self.obs = np.zeros((num_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(num_steps, dtype=np.int64)
        self.logprobs = np.zeros(num_steps, dtype=np.float32)
        self.rewards = np.zeros(num_steps, dtype=np.float32)
        self.dones = np.zeros(num_steps, dtype=np.float32)
        self.values = np.zeros(num_steps, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--lr-attacker", type=float, default=1e-4)
    parser.add_argument("--lr-defender", type=float, default=3e-4)
    parser.add_argument("--ent-coef-att", type=float, default=0.005)
    parser.add_argument("--ent-coef-def", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="models_multi")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiAgentHoneypotEnv()
    obs_dict, info = env.reset()

    obs_att = obs_dict["attacker"]
    obs_def = obs_dict["defender"]

    obs_att_flat = env.flatten_attacker_obs(obs_att)
    obs_def_flat = env.flatten_defender_obs(obs_def)

    mask_att = obs_att["action_mask"]

    obs_dim_att = obs_att_flat.shape[0]
    obs_dim_def = obs_def_flat.shape[0]
    mask_dim_att = mask_att.shape[0]

    n_actions_att = env.action_spaces["attacker"].n
    n_actions_def = env.action_spaces["defender"].n

    attacker = AttackerPPOAgent(obs_dim_att, n_actions_att, env.N_max, env.F_attacker).to(device)
    defender = DefenderPPOAgent(obs_dim_def, n_actions_def, env.N_max, env.F_defender).to(device)

    opt_att = optim.Adam(attacker.parameters(), lr=args.lr_attacker)
    opt_def = optim.Adam(defender.parameters(), lr=args.lr_defender)

    buf_att = AttackerRolloutBuffer(args.num_steps, obs_dim_att, mask_dim_att)
    buf_def = DefenderRolloutBuffer(args.num_steps, obs_dim_def)

    global_step = 0
    start_time = time.time()

    episode_returns_att = []
    episode_returns_def = []
    ep_ret_att = 0.0
    ep_ret_def = 0.0

    while global_step < args.total_timesteps:
        for t in range(args.num_steps):
            obs_att_flat = env.flatten_attacker_obs(obs_att)
            obs_def_flat = env.flatten_defender_obs(obs_def)
            mask_att = obs_att["action_mask"]

            buf_att.obs[t] = obs_att_flat
            buf_att.masks[t] = mask_att
            buf_def.obs[t] = obs_def_flat

            att_obs_tensor = torch.tensor(obs_att_flat, dtype=torch.float32, device=device).unsqueeze(0)
            att_mask_tensor = torch.tensor(mask_att, dtype=torch.bool, device=device).unsqueeze(0)

            def_obs_tensor = torch.tensor(obs_def_flat, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                a_def, logp_def, _, v_def = defender.get_action_and_value(def_obs_tensor)
                a_att, logp_att, _, v_att = attacker.get_action_and_value(att_obs_tensor, action_mask=att_mask_tensor)

            act_def = int(a_def.item())
            act_att = int(a_att.item())

            actions = {"defender": act_def, "attacker": act_att}
            next_obs_dict, rewards_dict, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated

            r_att = float(rewards_dict["attacker"])
            r_def = float(rewards_dict["defender"])

            buf_att.actions[t] = act_att
            buf_att.logprobs[t] = float(logp_att.cpu().item())
            buf_att.rewards[t] = r_att
            buf_att.dones[t] = float(done)
            buf_att.values[t] = float(v_att.cpu().item())

            buf_def.actions[t] = act_def
            buf_def.logprobs[t] = float(logp_def.cpu().item())
            buf_def.rewards[t] = r_def
            buf_def.dones[t] = float(done)
            buf_def.values[t] = float(v_def.cpu().item())

            ep_ret_att += r_att
            ep_ret_def += r_def

            global_step += 1
            obs_dict = next_obs_dict
            obs_att = obs_dict["attacker"]
            obs_def = obs_dict["defender"]

            if done:
                episode_returns_att.append(ep_ret_att)
                episode_returns_def.append(ep_ret_def)
                ep_ret_att = 0.0
                ep_ret_def = 0.0

                obs_dict, info = env.reset()
                obs_att = obs_dict["attacker"]
                obs_def = obs_dict["defender"]

            if global_step >= args.total_timesteps:
                break

        obs_att_flat = env.flatten_attacker_obs(obs_att)
        obs_def_flat = env.flatten_defender_obs(obs_def)
        mask_att = obs_att["action_mask"]

        att_obs_tensor = torch.tensor(obs_att_flat, dtype=torch.float32, device=device).unsqueeze(0)
        att_mask_tensor = torch.tensor(mask_att, dtype=torch.bool, device=device).unsqueeze(0)
        def_obs_tensor = torch.tensor(obs_def_flat, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            _, _, _, next_v_att = attacker.get_action_and_value(att_obs_tensor, action_mask=att_mask_tensor)
            _, _, _, next_v_def = defender.get_action_and_value(def_obs_tensor)

        next_v_att = float(next_v_att.cpu().item())
        next_v_def = float(next_v_def.cpu().item())

        adv_att = np.zeros_like(buf_att.rewards)
        last_gae = 0.0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                next_nonterm = 1.0 - buf_att.dones[t]
                next_value = next_v_att
            else:
                next_nonterm = 1.0 - buf_att.dones[t + 1]
                next_value = buf_att.values[t + 1]

            delta = (buf_att.rewards[t] + args.gamma * next_value * next_nonterm - buf_att.values[t])
            adv_att[t] = last_gae = (delta + args.gamma * args.gae_lambda * next_nonterm * last_gae)

        ret_att = adv_att + buf_att.values

        adv_def = np.zeros_like(buf_def.rewards)
        last_gae = 0.0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                next_nonterm = 1.0 - buf_def.dones[t]
                next_value = next_v_def
            else:
                next_nonterm = 1.0 - buf_def.dones[t + 1]
                next_value = buf_def.values[t + 1]

            delta = (buf_def.rewards[t] + args.gamma * next_value * next_nonterm - buf_def.values[t])
            adv_def[t] = last_gae = (delta + args.gamma * args.gae_lambda * next_nonterm * last_gae)

        ret_def = adv_def + buf_def.values

        b_obs_att = torch.tensor(buf_att.obs, dtype=torch.float32, device=device)
        b_actions_att = torch.tensor(buf_att.actions, dtype=torch.long, device=device)
        b_logprobs_att = torch.tensor(buf_att.logprobs, dtype=torch.float32, device=device)
        b_masks_att = torch.tensor(buf_att.masks, dtype=torch.bool, device=device)
        b_returns_att = torch.tensor(ret_att, dtype=torch.float32, device=device)
        b_adv_att = torch.tensor(adv_att, dtype=torch.float32, device=device)

        b_adv_att = (b_adv_att - b_adv_att.mean()) / (b_adv_att.std() + 1e-8)

        batch_size = args.num_steps
        minibatch = batch_size // 4
        idxs = np.arange(batch_size)

        for _ in range(args.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch):
                end = start + minibatch
                mb = idxs[start:end]

                logits, values = attacker(b_obs_att[mb], action_mask=b_masks_att[mb])
                dist = Categorical(logits=logits)

                new_logprob = dist.log_prob(b_actions_att[mb])
                entropy = dist.entropy().mean()

                ratio = (new_logprob - b_logprobs_att[mb]).exp()
                s1 = ratio * b_adv_att[mb]
                s2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * b_adv_att[mb]
                policy_loss = -torch.min(s1, s2).mean()

                value_loss = ((values - b_returns_att[mb]) ** 2).mean()

                loss = (policy_loss + args.vf_coef * value_loss - args.ent_coef_att * entropy)

                opt_att.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(attacker.parameters(), args.max_grad_norm)
                opt_att.step()

        b_obs_def = torch.tensor(buf_def.obs, dtype=torch.float32, device=device)
        b_actions_def = torch.tensor(buf_def.actions, dtype=torch.long, device=device)
        b_logprobs_def = torch.tensor(buf_def.logprobs, dtype=torch.float32, device=device)
        b_returns_def = torch.tensor(ret_def, dtype=torch.float32, device=device)
        b_adv_def = torch.tensor(adv_def, dtype=torch.float32, device=device)

        b_adv_def = (b_adv_def - b_adv_def.mean()) / (b_adv_def.std() + 1e-8)

        for epoch in range(args.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch):
                end = start + minibatch
                mb = idxs[start:end]

                logits, values = defender(b_obs_def[mb])
                dist = Categorical(logits=logits)

                new_logprob = dist.log_prob(b_actions_def[mb])
                entropy = dist.entropy().mean()

                ratio = (new_logprob - b_logprobs_def[mb]).exp()
                s1 = ratio * b_adv_def[mb]
                s2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * b_adv_def[mb]
                policy_loss = -torch.min(s1, s2).mean()

                value_loss = ((values - b_returns_def[mb]) ** 2).mean()

                loss = (policy_loss + args.vf_coef * value_loss - args.ent_coef_def * entropy)

                opt_def.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(defender.parameters(), args.max_grad_norm)
                opt_def.step()

        mean_ret_att = float(np.mean(episode_returns_att[-10:])) if episode_returns_att else 0.0
        mean_ret_def = float(np.mean(episode_returns_def[-10:])) if episode_returns_def else 0.0

        elapsed = time.time() - start_time
        # print(
        #     f"[step={global_step:8d}] "
        #     f"mean_ret_att={mean_ret_att:7.3f} "
        #     f"mean_ret_def={mean_ret_def:7.3f} "
        #     f"time={elapsed:6.1f}s"
        # )

        torch.save(
            {
                "model_state_dict": attacker.state_dict(),
                "obs_dim": obs_dim_att,
                "n_actions": n_actions_att,
                "N_max": env.N_max,
                "F_features": env.F_attacker,
            },
            os.path.join(args.save_dir, "att_temp.pt"),
        )

        torch.save(
            {
                "model_state_dict": defender.state_dict(),
                "obs_dim": obs_dim_def,
                "n_actions": n_actions_def,
                "N_max": env.N_max,
                "F_features": env.F_defender,
            },
            os.path.join(args.save_dir, "def_temp.pt"),
        )

    # print(f"Training done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
