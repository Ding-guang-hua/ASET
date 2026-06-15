import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
from data import CoS_Dataset
from RL import CosStateEncoder,PolicyNetwork,ProxyRewardModel,RL_Agent
from calculate import compute_distinct, compute_sim, compute_distinct_drug
from params import args
import json
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CosRL:

    def __init__(self, data):
        self.data = data

  
        self.model = RL_Agent(self.data).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def run(self):

        relation_adj = self.data.relation_adjs
        relEmbeddings = self.data.rel_embeddings
        rel_ids = self.data.rel_ids
        distinct_req = []
        for rid in rel_ids:
            adj = relation_adj[rid]
            dist_score = compute_distinct(adj)
            distinct_req.append(dist_score)
        print("", distinct_req)
        self.distinct_req = distinct_req
        all_batch_results = []
        for epoch in range(args.epoch):
            total_loss = 0.0
            total_reward = 0.0


            for rel_id in rel_ids:
                # print(rel_id,"-----------")
                self.optimizer.zero_grad()

                rel_embed = torch.tensor(relEmbeddings[rel_id - 1], dtype=torch.float32).to(device)
                loss, reward = self.model.train_step(rel_embed, rel_id, relation_adj, self.distinct_req)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_reward += reward

            avg_loss = total_loss / len(rel_ids)
            avg_reward = total_reward / len(rel_ids)
            print(f'Train Epoch: {epoch + 1}/{args.epoch} | Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f}')

            best_cos = self.calibrate_and_output(relation_adj, rel_ids)
            print(f'Test Epoch {epoch + 1}/{args.epoch} | CoS:{best_cos}')

            result_item = {
                "epoch": epoch + 1,
                "rel_seq": best_cos['sequence'],
                "reward": best_cos['reward']
            }
            all_batch_results.append(result_item)

        all_batch_results.sort(key=lambda x: x["reward"], reverse=True)
        save_path = self.data.predir + args.data + '_CoS.json'
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_batch_results, f, indent=4, ensure_ascii=False)
        best_epoch = all_batch_results[0]['epoch']
        best_req = all_batch_results[0]['rel_seq']
        best_reward = all_batch_results[0]['reward']
        print()
        print(f'Best Epoch {best_epoch} | CoS:{best_req} | Reward: {best_reward}')

    def calibrate_and_output(self, relation_adjs, rel_ids):

        self.model.policy_net.eval()
        self.model.state_encoder.eval()
        self.model.proxy_reward.eval()
        candidate_cos = []

        with torch.no_grad():

            for _ in range(10):

                initial_rel = rel_ids[torch.randint(0, len(rel_ids), ()).item()]
                r_seq = [int(initial_rel)]
                total_reward = 0.0  
                reward_list = []


                for step in range(len(rel_ids)):

                    state = self.model.state_encoder(r_seq, relation_adjs,self.distinct_req)
                    
                    action_probs, _ = self.model.policy_net(state)

                   
                    available_actions = [r for r in rel_ids if r not in r_seq]
                    if len(available_actions) == 0:
                        break

                    
                    available_probs = action_probs[0, [a - 1 for a in available_actions]]
                    best_idx = torch.argmax(available_probs).item()
                    selected_rel = available_actions[best_idx]

                    distinct_r = self.distinct_req[selected_rel-1]
                    if len(r_seq) >= 1:
                        sim_r_prev = compute_sim(relation_adjs[r_seq[-1]], relation_adjs[selected_rel])
                    else:
                        sim_r_prev = 0.0
                    features = torch.tensor([[distinct_r, sim_r_prev]], dtype=torch.float32, device=device)
                    reward_input = torch.cat([state, features], dim=1)
                    reward = self.model.proxy_reward(reward_input)
                    
                    total_reward += reward.item()

                   
                    r_seq.append(int(selected_rel))
                    reward_list.append(reward.item())

                
                candidate_cos.append({
                    "sequence": r_seq,
                    "reward": total_reward / len(reward_list) if len(reward_list) > 0 else 0.0
                })

        
        candidate_cos.sort(key=lambda x: x["reward"], reverse=True)

        return candidate_cos[0]

if __name__ == '__main__':
    torch.cuda.set_device(args.gpu)
    decice_idx = torch.cuda.current_device()
    print(f"GPU---{decice_idx}")
    data = CoS_Dataset()
    data.load_data()

    coach = CosRL(data)
    coach.run()
