from kaggle_environments import make
from tqdm import tqdm
import torch
from torch.nn import functional as F
import numpy as np
import itertools
import random
import math
from torch.utils.tensorboard import SummaryWriter
from model import ParseState, Policy
import argparse
import torch.multiprocessing as mp

def discounted_rewards(rewards, gamma):
    out = []
    acc = 0.0
    for r in rewards[::-1]:
        acc = r + gamma * acc
        out.insert(0, acc)
    return out

def train_epoch(trainer, policy, optimizer, device):
    optimizer.zero_grad()
    batch_size = 16

    episode_log_probs = []
    episode_entropy = []
    pbar = tqdm(desc='Train Batch', total=batch_size, leave=False)


    # with mp.Pool(8) as p:
    #     print(p.starmap(play_episode, [(policy, device,)] * batch_size))

    train_queue = mp.Queue(maxsize=8)
    processes = []


    for _ in range(8):
        p = mp.Process(target=play_episode, args=(policy, device, train_queue))
        p.start()
        processes.append(p)
    
    
    for _ in range(batch_size):
        train_queue.get()
        # episode_log_probs.append(lp)
        # episode_entropy.append(ent)
        pbar.update()
    
    for p in processes:
        p.terminate()
        p.join()

    pbar.clear()
    pbar.close()

    # loss = -1.0 * torch.stack(episode_log_probs).mean()
    # loss += 1e-2 * torch.stack(episode_entropy).mean()
    # loss.backward()
    # optimizer.step()

def play_episode(policy, device, queue):
    env = make("halite", debug=True)
    trainer = env.train([None, "random", "random", "random"])
    with torch.no_grad():
        while True:
            
            s = trainer.reset()

            actions = []
            probs = []
            rewards = []
            base = []
            masks = []

            terminal = False
            while not terminal:
                a, l_a, r_a, m = policy.action(s)
                s_, r, terminal, info = trainer.step(a)
                
                rewards.append(r)
                base.append(sum(rewards)/len(rewards))
                actions.append(r_a.detach())
                # probs.append(l_a)
                # masks.append(m)

                s = s_
            queue.put((rewards, base, actions))
            continue 
            probs = torch.cat(probs) + 1e-7
            actions = F.one_hot(torch.stack(actions), 7)
            mask = torch.stack(masks).float()


            masked_log_prob = torch.log(probs) * actions * mask.unsqueeze(-1)
            # (T, E, A)

            masked_log_prob = torch.sum(masked_log_prob, (1,2))
            
            rewards = discounted_rewards(rewards, .99)
            rewards = torch.tensor(rewards).float().to(device)
            rewards = rewards - torch.tensor(base).float().to(device)

            weighted_log_prob = rewards * masked_log_prob
            weighted_log_prob = torch.sum(weighted_log_prob, -1)

            queue.put(rewards.detach())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--players", default=4, type=int,
                        help="Number of players in halite [default: 4]")
    parser.add_argument("--n_episodes", default=32, type=int,
                        help="Batch size [default: 32]")
    # parser.add_argument("--lr", default=1e-3, type=float,
    #                     help="Learning rate [default: 20]")
    # parser.add_argument("--epoch", default=100, type=int,
    #                     help="Number of training epochs [default: 100]")
    # parser.add_argument("--device", type=int,
    #                     help="GPU card ID to use (if not given, use CPU)")
    # parser.add_argument("--seed", default=42, type=int,
    #                     help="Random seed [default: 42]")
    args = parser.parse_args()
    
    env = make("halite", debug=True)

    print(env.configuration)

    # Training agent in first position (player 1) against the default random agent.
    trainer = env.train([None, "random", "random", "random"])

   
    device = 'cuda:0'
    policy = Policy(env.configuration, args.players).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2, weight_decay=0)

    policy.share_memory()
    torch.multiprocessing.set_start_method('spawn')

    n_epochs = 50
    pbar = tqdm(desc='Train Epoch', total=n_epochs, leave=False)
    for i in range(n_epochs):
        train_epoch(trainer, policy, optimizer, device)
        pbar.update()

        f = open('./index.html', 'w')
        f.write(env.render(mode='html', width=400, height=600))
        f.close()
    
    pbar.clear()
    pbar.close()
   