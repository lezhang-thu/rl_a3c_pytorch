from __future__ import division
# from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Clstm
from player_util import Agent
import time
import logging
from utils import weights_init


def test(args, shared_model, env_conf):
    # ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    device = torch.device('cuda:{}'.format(gpu_id) if gpu_id >= 0 else 'cpu')

    log = {}
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    env = atari_env(args.env, env_conf, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None, gpu_id=gpu_id)
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)
    player.model.apply(weights_init)

    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).to(torch.float32)

    player.model = player.model.to(device)
    player.state = player.state.to(device)

    flag = True
    max_score = 0
    while True:
        if flag:
            player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False

        player.action_test()
        reward_sum += player.reward

        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).to(torch.float32)
            player.state = player.state.to(device)
        elif player.info:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, '{0}{1}.dat'.format(
                    args.save_model_dir, args.env))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)

            player.state = torch.from_numpy(state).to(torch.float32)
            player.state = player.state.to(device)
