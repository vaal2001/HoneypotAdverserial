import argparse
import json
import os
import numpy as np
import torch

from HoneypotAttacker.env.env import HoneypotDetectionEnv
from HoneypotAttacker.rl.ppo_agent import PPOAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument( "--log-path", type=str, default="logs/attacker_ppo_latest_logs.jsonl", help="Where to store step-by-step evaluation logs (JSONL)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)

    obs_dim = ckpt["obs_dim"]
    n_actions = ckpt["n_actions"]
    N_max = ckpt["N_max"]
    F_features = ckpt["F_features"]

    env = HoneypotDetectionEnv(N_max=N_max, F=F_features)
    agent = PPOAgent(obs_dim, n_actions, N_max, F_features).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    total_correct = 0
    total_wrong = 0
    total_classified = 0
    total_hosts_seen = 0
    returns = []

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    print("Evaluating classification accuracy...\n")

    with open(args.log_path, "w") as f_log:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            mask = obs["action_mask"]
            obs_flat = env.flatten_obs(obs)
            done = False

            ep_correct = 0
            ep_wrong = 0
            ep_classified = 0
            ret = 0.0
            total_hosts_seen += env.N_actual
            step_idx = 0

            while not done:
                obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
                mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

                with torch.no_grad():
                    a, _, _, _ = agent.get_action_and_value(obs_tensor, action_mask=mask_tensor)

                action = a.item()
                host = action // env.K
                sub = action % env.K

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ret += reward

                if sub in (5, 6) and 0 <= host < env.N_actual:
                    ep_classified += 1
                    true_is_honey = (env.hosts[host].host_type.name == "HONEYPOT")
                    predicted_honey = (sub == 6)

                    if predicted_honey == true_is_honey:
                        ep_correct += 1
                        if true_is_honey:
                            TP += 1
                        else:
                            TN += 1
                    else:
                        ep_wrong += 1
                        if predicted_honey:
                            FP += 1
                        else:
                            FN += 1

                record = {
                    "episode": ep,
                    "step": step_idx,
                    "action": int(action),
                    "host": int(host),
                    "sub_action": int(sub),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": bool(done),
                    "N_actual": int(env.N_actual),
                    "mask_known": env.mask_known[:env.N_actual].astype(int).tolist(),
                    "mask_classified": env.mask_classified[:env.N_actual].astype(int).tolist(),
                    "classified_as": [None if c is None else bool(c) for c in env.classified_as[:env.N_actual]],
                    "true_labels": [(h.host_type.name == "HONEYPOT") for h in env.hosts[:env.N_actual]],
                    "obs_flat": obs_flat.tolist(),
                    "action_mask": mask.tolist(),
                }
                f_log.write(json.dumps(record) + "\n")

                step_idx += 1
                obs = next_obs
                obs_flat = env.flatten_obs(next_obs)
                mask = next_obs["action_mask"]

            total_correct += ep_correct
            total_wrong += ep_wrong
            total_classified += ep_classified
            returns.append(ret)

            print(
                f"Ep {ep+1}: return={ret:.2f}, class={ep_classified}/{env.N_actual}, "
                f"correct={ep_correct}, wrong={ep_wrong}"
            )

    accuracy = total_correct / max(1, total_classified)

    precision = TP / max(1, (TP + FP))
    recall = TP / max(1, (TP + FN))
    f1 = 2 * precision * recall / max(1e-6, (precision + recall))

    specificity = TN / max(1, (TN + FP))
    false_positive_rate = FP / max(1, (FP + TN))
    false_negative_rate = FN / max(1, (TP + FN))
    balanced_accuracy = (recall + specificity) / 2

    print("\n======== FINAL CLASSIFICATION REPORT ========")
    print(f"Episodes: {args.episodes}")
    print(f"Avg return: {np.mean(returns):.2f}")
    print(f"Total hosts: {total_hosts_seen}")
    print(f"Classified: {total_classified}")
    print(f"Correct: {total_correct}")
    print(f"Wrong: {total_wrong}")
    print(f"Accuracy: {accuracy:.3f}")
    print("")
    print("---- Confusion Matrix ----")
    print(f"TP (honeypot-honeypot):{TP}")
    print(f"TN (real-real): {TN}")
    print(f"FP (real-honeypot): {FP}")
    print(f"FN (honeypot-real): {FN}")
    print("")
    print("---- Advanced Metrics ----")
    print(f"Precision: {precision:.3f}")
    print(f"Recall (TPR): {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"Specificity (TNR): {specificity:.3f}")
    print(f"FPR: {false_positive_rate:.3f}")
    print(f"FNR: {false_negative_rate:.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.3f}")
    print("")
    print(f"Step log: {args.log_path}")
    print("=============================================")

if __name__ == "__main__":
    main()
