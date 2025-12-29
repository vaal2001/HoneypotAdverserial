import argparse
import torch
import numpy as np

from HoneypotArena.multi_agent_env.multi_env import MultiAgentHoneypotEnv
from HoneypotAttacker.rl.ppo_agent import PPOAgent as AttackerPPOAgent
from HoneypotDefender.rl.ppo_agent import PPOAgent as DefenderPPOAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacker-model", type=str, required=True)
    parser.add_argument("--defender-model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiAgentHoneypotEnv()
    obs_dict, _ = env.reset()

    att_ckpt = torch.load(args.attacker_model, map_location=device, weights_only=False)
    obs_dim_att = att_ckpt["obs_dim"]
    n_actions_att = att_ckpt["n_actions"]
    N_max_att = att_ckpt["N_max"]
    F_att = att_ckpt["F_features"]

    attacker = AttackerPPOAgent(obs_dim_att, n_actions_att, N_max_att, F_att).to(device)

    attacker.load_state_dict(att_ckpt["model_state_dict"])
    attacker.eval()

    def_ckpt = torch.load(args.defender_model, map_location=device, weights_only=False)
    obs_dim_def = def_ckpt["obs_dim"]
    n_actions_def = def_ckpt["n_actions"]
    N_max_def = def_ckpt["N_max"]
    F_def = def_ckpt["F_features"]

    defender = DefenderPPOAgent(obs_dim_def, n_actions_def, N_max_def, F_def).to(device)

    defender.load_state_dict(def_ckpt["model_state_dict"])
    defender.eval()

    all_att_returns = []
    all_def_returns = []

    total_correct = 0
    total_wrong = 0
    total_classified = 0

    for ep in range(args.episodes):

        obs_dict, _ = env.reset()
        obs_att = obs_dict["attacker"]
        obs_def = obs_dict["defender"]

        done = False
        ep_ret_att = 0.0
        ep_ret_def = 0.0

        ep_correct = 0
        ep_wrong = 0
        ep_classified = 0

        print(f"\n=== EPISODE {ep + 1} ===")

        while not done:

            att_flat = env.flatten_attacker_obs(obs_att)
            att_mask = obs_att["action_mask"]

            att_tensor = torch.tensor(att_flat, dtype=torch.float32, device=device).unsqueeze(0)

            mask_tensor = torch.tensor(att_mask, dtype=torch.bool, device=device).unsqueeze(0)

            def_flat = env.flatten_defender_obs(obs_def)
            def_tensor = torch.tensor(def_flat, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                a_def, _, _, _ = defender.get_action_and_value(def_tensor)
                a_att, _, _, _ = attacker.get_action_and_value(att_tensor, action_mask=mask_tensor)

            def_action = int(a_def.item())
            att_action = int(a_att.item())

            actions = {"defender": def_action, "attacker": att_action}

            next_obs_dict, rewards, terminated, truncated, _ = env.step(actions)

            r_att = rewards["attacker"]
            r_def = rewards["defender"]

            ep_ret_att += r_att
            ep_ret_def += r_def

            done = terminated or truncated

            host = att_action // env.K_attacker
            sub = att_action % env.K_attacker

            if sub in (5, 6) and 0 <= host < env.N_actual:
                predicted_honey = (sub == 6)
                true_honey = (env.hosts[host].host_type.name == "HONEYPOT")

                ep_classified += 1
                total_classified += 1

                if predicted_honey == true_honey:
                    ep_correct += 1
                    total_correct += 1
                else:
                    ep_wrong += 1
                    total_wrong += 1

            obs_att = next_obs_dict["attacker"]
            obs_def = next_obs_dict["defender"]

        all_att_returns.append(ep_ret_att)
        all_def_returns.append(ep_ret_def)

        if ep_classified > 0:
            ep_accuracy = ep_correct / ep_classified
        else:
            ep_accuracy = 0.0

        print(f"Attacker return: {ep_ret_att:.2f}")
        print(f"Defender return: {ep_ret_def:.2f}")
        print(f"Classified:      {ep_classified}")
        print(f"Correct:         {ep_correct}")
        print(f"Wrong:           {ep_wrong}")
        print(f"Accuracy:        {ep_accuracy:.3f}")

    print("\n================ FINAL RESULTS ================")
    print(f"Episodes: {args.episodes}")
    print(f"Average attacker return: {np.mean(all_att_returns):.3f}")
    print(f"Average defender return: {np.mean(all_def_returns):.3f}")

    print(f"Total classified: {total_classified}")
    print(f"Total correct:    {total_correct}")
    print(f"Total wrong:      {total_wrong}")

    if total_classified > 0:
        accuracy = total_correct / total_classified
        print(f"Overall accuracy: {accuracy:.3f}")
    else:
        print("Overall accuracy: N/A (no classifications made)")

if __name__ == "__main__":
    main()
