from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from utils import weights_init


def train(rank, args, shared_model, optimizer, env_conf):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    device = torch.device('cuda:{}'.format(gpu_id) if gpu_id >= 0 else 'cpu')

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None, gpu_id=gpu_id)

    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)
    player.model.apply(weights_init)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).to(torch.float32)
    player.state = player.state.to(device)
    player.model = player.model.to(device)

    player.model.train()
    player.eps_len += 2
    while True:
        player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            player.cx = torch.zeros(1, 512).to(device)
            player.hx = torch.zeros(1, 512).to(device)
        else:
            player.cx = player.cx.detach()
            player.hx = player.hx.detach()

        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).to(torch.float32)
            player.state = player.state.to(device)

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model((player.state.unsqueeze(0),
                                        (player.hx, player.cx)))
            R = value.detach()

        R = R.to(device)

        player.values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).to(device)

        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * player.values[i + 1].detach(
            ) - player.values[i].detach()

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - player.log_probs[i] * gae - 0.01 * player.entropies[i]

        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()
