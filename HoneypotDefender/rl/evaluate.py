from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from HoneypotDefender.env.env import HoneypotDefenderEnv
from HoneypotDefender.rl.ppo_agent import PPOAgent
from HoneypotDefender.env.kl_metrics import kl_real_vs_honeypot
from HoneypotDefender.env.detectability import detectability_single


def serialize_obs(env, obs_flat):
    """
    Simpele serialisatie van de observatie.
    Wil je meer details loggen (host params etc.), kun je dat hier uitbreiden.
    """
    return {
        "obs_flat": obs_flat.tolist(),
    }


def heuristic_attack_metrics(env: HoneypotDefenderEnv) -> tuple[float, float]:
    """
    Heuristische 'attacker' op basis van detectability_single.

    We berekenen per host een detectability-score t.o.v. de REAL-baseline,
    kiezen een vaste threshold en classificeren:

        score >= tau  → voorspeld HONEYPOT
        score <  tau  → voorspeld REAL

    ASR  = (TP + TN) / N
    HDR  = TP / (TP + FN)
    """
    hosts = env.hosts
    N_actual = env.N_actual
    adj = env.adjacency
    baseline = env.baseline_stats

    if baseline is None or N_actual == 0:
        return 0.0, 0.0

    deg = adj[:N_actual, :N_actual].sum(axis=1).astype(float)

    scores = []
    labels = []

    for i, h in enumerate(hosts[:N_actual]):
        s = detectability_single(h, degree=deg[i], baseline=baseline)
        scores.append(s)
        labels.append(h.host_type == h.host_type.__class__.HONEYPOT)

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    # simpele vaste threshold (kan je later tunen of adaptief maken)
    tau = 3.0

    preds = scores >= tau  # True = predicted honeypot

    # Confusion matrix
    true_honey = labels
    true_real = ~labels

    TP = np.logical_and(preds, true_honey).sum()
    FN = np.logical_and(~preds, true_honey).sum()
    TN = np.logical_and(~preds, true_real).sum()
    FP = np.logical_and(preds, true_real).sum()

    N = max(1, N_actual)
    asr = float((TP + TN) / N)
    hdr = float(TP / max(1, (TP + FN)))

    return asr, hdr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Pad naar getrainde PPO defender. Niet nodig bij --random.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/defender_eval_log.jsonl",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Gebruik random actions i.p.v. PPO policy (baseline).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # -------------------------
    # AGENT INIT
    # -------------------------
    agent = None
    if not args.random:
        if args.model_path is None:
            raise ValueError("You must provide --model-path when not using --random baseline.")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)

        ckpt_obs_dim = ckpt["obs_dim"]
        ckpt_n_actions = ckpt["n_actions"]
        ckpt_N_max = ckpt["N_max"]
        ckpt_F_features = ckpt["F_features"]

        assert ckpt_obs_dim == obs_dim, f"Checkpoint obs_dim={ckpt_obs_dim}, env obs_dim={obs_dim}"
        assert ckpt_n_actions == n_actions, f"Checkpoint n_actions={ckpt_n_actions}, env n_actions={n_actions}"
        assert ckpt_N_max == N_max, f"Checkpoint N_max={ckpt_N_max}, env N_max={N_max}"
        assert ckpt_F_features == F_features, f"Checkpoint F={ckpt_F_features}, env F={F_features}"

        agent = PPOAgent(obs_dim, n_actions, N_max, F_features).to(device)
        agent.load_state_dict(ckpt["model_state_dict"])
        agent.eval()

    returns = []
    all_detect_honey = []  # per-episode mean detectability honeypots
    all_detect_real = []   # per-episode mean detectability real hosts
    all_kl = []            # per-episode mean KL
    all_asr = []           # per-episode Attack Success Rate
    all_hdr = []           # per-episode Honey Detection Rate

    mode = "RANDOM" if args.random else "PPO"
    print(f"Starting Defender Evaluation... (mode={mode})\n")

    with open(args.log_path, "w") as f_log:

        for ep in range(args.episodes):
            obs, info = env.reset()
            obs_flat = env.flatten_obs(obs)
            done = False

            step_idx = 0
            ep_return = 0.0

            detect_honey_total = 0.0
            detect_real_total = 0.0
            kl_total = 0.0

            while not done:
                obs_tensor = torch.tensor(
                    obs_flat, dtype=torch.float32, device=device
                )

                if args.random:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        a, logp, ent, v = agent.get_action_and_value(obs_tensor)
                    action = int(a.item())

                # State before
                state_before = serialize_obs(env, obs_flat)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_return += reward

                # State after
                next_flat = env.flatten_obs(next_obs)
                state_after = serialize_obs(env, next_flat)

                # Update metrics (per step)
                detect_honey_total += env.last_mean_detect_honey
                detect_real_total += env.last_mean_detect_real

                # Compute KL per step
                kl = kl_real_vs_honeypot(env.hosts, env.adjacency, env.N_actual)
                if np.isnan(kl):
                    kl = 0.0
                kl_total += kl

                # Write JSONL record (step-level)
                record = {
                    "episode": ep,
                    "step": step_idx,
                    "action": int(action),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": bool(done),
                    "state_before": state_before,
                    "state_after": state_after,
                    "metrics": {
                        "detect_honey": float(env.last_mean_detect_honey),
                        "detect_real": float(env.last_mean_detect_real),
                        "KL": float(kl),
                    },
                }

                f_log.write(json.dumps(record) + "\n")

                # Prepare next input
                obs = next_obs
                obs_flat = next_flat

                step_idx += 1

            # Per-episode gemiddelden (detectability + KL)
            steps = max(1, step_idx)
            mean_detect_honey = detect_honey_total / steps
            mean_detect_real = detect_real_total / steps
            mean_kl = kl_total / steps

            # Heuristic attacker metrics (op eindconfiguratie)
            asr, hdr = heuristic_attack_metrics(env)

            returns.append(ep_return)
            all_detect_honey.append(mean_detect_honey)
            all_detect_real.append(mean_detect_real)
            all_kl.append(mean_kl)
            all_asr.append(asr)
            all_hdr.append(hdr)

            print(
                f"Episode {ep+1}: return={ep_return:.2f}, "
                f"ASR={asr:.3f}, HDR={hdr:.3f}, "
                f"detect_honey_mean={mean_detect_honey:.3f}, "
                f"detect_real_mean={mean_detect_real:.3f}, "
                f"KL_mean={mean_kl:.3f}"
            )

    print("\n===== FINAL EVAL =====")
    print(f"Mode:                  {mode}")
    print(f"Episodes:              {args.episodes}")
    print(f"Avg return:            {np.mean(returns):.3f}")
    print(f"Avg ASR:               {np.mean(all_asr):.3f}   # Attack Success Rate")
    print(f"Avg HDR:               {np.mean(all_hdr):.3f}   # Honey Detection Rate")
    print(f"Avg detect_honeypot:   {np.mean(all_detect_honey):.3f}  # per step")
    print(f"Avg detect_real:       {np.mean(all_detect_real):.3f}   # per step")
    print(f"Avg KL(real||honeypot):{np.mean(all_kl):.3f}")
    print(f"Step log:              {args.log_path}")


if __name__ == "__main__":
    main()
