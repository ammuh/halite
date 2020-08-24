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

class Expierence():
    def __init__(self, n_procs, policy):
        self.policy = policy
        self.q = mp.Queue(maxsize=n_procs)
        self.n_procs = n_procs

        self.processes = []
        for _ in range(self.n_procs):
            p = mp.Process(target=Expierence.play_episode, args=(policy, self.q))
            p.start()
            self.processes.append(p)
    
    @staticmethod
    def play_episode(policy, queue):
        env = make("halite", debug=True)
        agents = [None, "random", "random", "random"]
        with torch.no_grad():
            while True:
                agents = agents[:]
                random.shuffle(agents)
                trainer = env.train(agents)
                s = trainer.reset()
                states = []
                actions = []
                rewards = []
                terminal = False
                while not terminal:
                    a, r_a = policy.action(s)
                    s_, r, terminal, info = trainer.step(a[0])
                    states.append(s)
                    actions.append(r_a[0].detach().tolist())
                    del r_a
                    rewards.append(r)
                
                    s = s_
                queue.put((states, actions, rewards))
    
    def get_batch(self, n):
        states, actions, rewards = [], [], []

        pbar = tqdm(desc='Fetching Episode Batch', total=n, leave=False)
        for _ in range(n):
            s, a, r = self.q.get()
            states.append(s)
            actions.append(a)
            rewards.append(r)
            pbar.update()
        
        pbar.clear()
        pbar.close()

        return states, actions, rewards
    
    def restart(self):
        self.termintate()

        self.processes = []
        for _ in range(self.n_procs):
            p = mp.Process(target=Expierence.play_episode, args=(policy, self.q))
            p.start()
            self.processes.append(p)

    def termintate(self):
        for p in self.processes:
            p.terminate()
            p.join()

def get_baseline(rewards):
    b = []
    for i, r in enumerate(rewards):
        b.append(sum(rewards[0:i+1]) * 1.0 / (i+1))
    return b

def discounted_rewards(rewards, gamma):
    out = []
    acc = 0.0
    for r in rewards[::-1]:
        acc = r + gamma * acc
        out.insert(0, acc)
    return out

def train_epoch(epoch, policy, exp, optimizer, device, writer, args):
    optimizer.zero_grad()
    batch_size = args.batch_episodes

    states, actions, rewards = exp.get_batch(batch_size)


    sar = zip(states, actions, rewards)
    loss_avg = 0.0
    
    pbar = tqdm(desc='Accumulating Gradients', total=batch_size, leave=False)

    avg_rewards = 0
    for s, a, r in sar:
        probs, mask = policy(s, mask=True)
        probs = probs + 1e-7

        a = torch.tensor(a)
        actions = F.one_hot(a, 7).to(device)
        mask = mask.float()
        masked_log_prob = torch.log(probs) * actions * mask.unsqueeze(-1)
    
        masked_log_prob = torch.sum(masked_log_prob, (1,2))

        baseline = get_baseline(r)
        rewards = discounted_rewards(r, args.gamma)
        baseline = torch.tensor(baseline).float().to(device)
        rewards = torch.tensor(rewards).float().to(device)
        rewards = rewards - baseline

        weighted_log_prob = rewards * masked_log_prob
       
        weighted_log_prob = -1.0 * torch.mean(weighted_log_prob, -1)


        # entropy
        beta = args.beta
        entropy = beta * torch.mean(torch.sum(probs * torch.log(probs), (1,2)))
       
        loss = (weighted_log_prob + entropy)/batch_size
        loss_avg += loss.item()
        loss.backward()
        pbar.update()
        avg_rewards += sum(r)*1.0 / batch_size
    
    nn.utils.clip_grad_norm_(policy.parameters(), args.clip)
    optimizer.step()
    
    pbar.clear()
    pbar.close()

    writer.add_scalar('Loss', loss_avg, epoch)
    writer.add_scalar('Rewards', avg_rewards, epoch)


    for s, a, r in sar:
        for s_ in s:
            del s_
        for a_ in a:
            del a_
        for r_ in r:
            del r_
    del sar
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--players", default=4, type=int,
                        help="Number of players in halite [default: 4]")
    parser.add_argument("--batch_episodes", default=128, type=int,
                        help="Batch size [default: 64]")
    parser.add_argument("--gamma", default=.99, type=float,
                        help="Gamma value [default: .99]")
    parser.add_argument("--beta", default=1e-2, type=float,
                        help="Entropy Beta value [default: 1e-2]")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate [default: 1e-3]")
    parser.add_argument("--clip", default=1e-1, type=float,
                        help="Grad clip [default: 1e-1]")
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
   
    device = 'cuda:0'
    policy = Policy(env.configuration, args.players).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=0)

    policy.share_memory()
    torch.multiprocessing.set_start_method('spawn')

    exp = Expierence(8, policy)

    writer = SummaryWriter()

    n_epochs = 100
    try:
        pbar = tqdm(desc='Train Epoch', total=n_epochs, leave=False)
        for i in range(n_epochs):
            train_epoch(i, policy, exp, optimizer, device, writer, args)
            pbar.update()

        pbar.clear()
        pbar.close()
    finally:
        exp.termintate()
    
   