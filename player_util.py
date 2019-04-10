from __future__ import division
import torch
import torch.nn.functional as F


class Agent(object):
    def __init__(self, model, env, args, state, *, gpu_id=-1):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0

        self.device = torch.device('cuda:{}'.format(gpu_id)
                                   if gpu_id >= 0 else 'cpu')

    def action_train(self):
        value, logit, (self.hx, self.cx) = self.model((self.state.unsqueeze(0),
                                                       (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).detach()
        log_prob = log_prob.gather(1, action)
        state, self.reward, self.done, self.info = self.env.step(action.item())

        self.state = torch.from_numpy(state).to(torch.float32)
        self.state = self.state.to(self.device)

        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                self.cx = torch.zeros(1, 512).to(self.device)
                self.hx = torch.zeros(1, 512).to(self.device)
            else:
                self.cx = self.cx.detach()
                self.hx = self.hx.detach()
            value, logit, (self.hx, self.cx) = self.model(
                (self.state.unsqueeze(0), (self.hx, self.cx)))
            prob = F.softmax(logit, dim=1)
            action = torch.max(prob, 1)

        action = action.item()
        state, self.reward, self.done, self.info = self.env.step(action)

        self.state = torch.from_numpy(state).to(torch.float32)
        self.state = self.state.to(self.device)

        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
