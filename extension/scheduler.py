import math
from bisect import bisect_right
from typing import List

from torch.optim.optimizer import Optimizer

from extension.config import get_cfg, get_parser
from extension.logger import get_logger
from extension.utils import str2list, extend_list


class FixLR:
    def __init__(self, steps, start_lr, end_lr):
        self.lr = end_lr

    def __call__(self, step):
        return self.lr


class StepLR:
    def __init__(self, steps, start_lr, end_lr):
        self.gamma = (end_lr - start_lr) / steps
        self.lr = start_lr

    def __call__(self, step):
        return self.lr + self.gamma * step


class ExpLR:
    def __init__(self, steps, start_lr, end_lr):
        self.lr = start_lr
        self.gamma = (end_lr / start_lr) ** (1 / steps)

    def __call__(self, step):
        return self.lr * self.gamma ** step


class CosineLR:
    def __init__(self, steps, start_lr, end_lr):
        self.lr = end_lr
        self.gamma = 0.5 * (start_lr - end_lr)
        self.beta = math.pi / steps

    def __call__(self, step):
        return self.lr + self.gamma * (1 + math.cos(step * self.beta))


_lr_methods = {
    'fix': FixLR,
    'step': StepLR,
    'exp': ExpLR,
    'cos': CosineLR,
}


class LRScheduler(object):
    def __init__(self, optimizer: Optimizer, methods: List[str], steps: List[int], lr_starts: List[float],
                 lr_ends: List[float], num_steps=1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self._optimizer = optimizer
        self.num_steps = num_steps
        self.BIAS_LR_FACTOR = get_cfg("bias_lr_factor")
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self._base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        num = max(len(steps), len(lr_ends), len(methods))
        self._steps = extend_list(steps, num)
        self._lr_ends = extend_list(lr_ends, num)
        self._lr_starts = [1.] + self._lr_ends[:num] if lr_starts is None else extend_list(lr_starts, num)
        self._starts = [0]
        methods = extend_list(methods, num)
        self._methods = []
        for i in range(num):
            assert methods[i] in _lr_methods, 'No such LR scheduler {}'.format(methods[i])
            self._methods.append(
                _lr_methods[methods[i]](self._steps[i] * num_steps, self._lr_starts[i], self._lr_ends[i]))
        for step in steps:
            assert step >= 0, 'The step must be greater than 0'
            self._starts.append(self._starts[-1] + step)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self, base_lr=1, epoch=0, step=0):
        idx = bisect_right(self._starts, epoch, hi=len(self._methods)) - 1
        epoch = epoch - self._starts[idx]
        return base_lr * self._methods[idx](epoch * self.num_steps + step)

    def step(self, epoch=0, step=0):
        for param_group, base_lr in zip(self._optimizer.param_groups, self._base_lrs):
            if param_group['weight_decay'] == 0:
                param_group['lr'] = self.get_lr(base_lr, epoch, step) * self.BIAS_LR_FACTOR #if no_bias_wd, check the bias lr factor
            else:
                param_group['lr'] = self.get_lr(base_lr, epoch, step)

    def __repr__(self):
        fs = "\n"
        step_len = len(str(self._starts[-1]))
        base_lr = self._base_lrs[0]
        fmt_str = '\t{:10s} change lr: {:.1e}-->{:.1e} at epoch [{:%dd}, {:%dd}) in {:%dd} epochs\n' % (
            step_len, step_len, step_len)
        for i in range(len(self._methods)):
            if self._steps[i] == 0:
                continue
            fs += fmt_str.format(
                self._methods[i].__class__.__name__,
                self._lr_starts[i]*base_lr,
                self._lr_ends[i]*base_lr,
                self._starts[i],
                self._starts[i + 1],
                self._steps[i],
            )
        fs += '\t steps in one epoch: {}'.format(self.num_steps)
        return fs

    def draw(self, total_epochs=100, log_scale=True, lr=1):
        import matplotlib.pyplot as plt
        lrs = []
        x = []
        for i in range(total_epochs):
            for step in range(self.num_steps):
                x.append(i + step / self.num_steps)
                lrs.append(self.get_lr(lr, i, step))
        ax = plt.gca()
        if log_scale:
            ax.set_yscale('log')
        plt.plot(x, lrs)
        # plt.xticks(list(range(total_epochs)))
        plt.show()


def options(parser=None):
    if parser is None:
        parser = get_parser()
    # train learning rate
    _methods = list(_lr_methods.keys())
    group = parser.add_argument_group("Learning rate scheduler Option:")
    group.add_argument("--lr", default=0.1, type=float, metavar="V", help="The base learning rate.")
    group.add_argument("--lr-methods", default=['fix'], type=str2list, metavar="C",
                       help="The learning rate scheduler: {" + "\n".join(_methods) + "}")
    group.add_argument("--lr-starts", default=None, type=str2list, metavar='Vs',
                       help="The start learning rate factor for each stage.")
    group.add_argument("--lr-ends", default=[1.], type=str2list, metavar='Vs',
                       help="The end learning rate factor for each stage.")
    group.add_argument("--lr-steps", default=[100], type=str2list, metavar='Ns',
                       help='the step values for learning rate policy "steps"')
    return group


def make(optimizer: Optimizer, cfg, num_steps=1):
    scheduler = LRScheduler(optimizer, cfg.lr_methods, cfg.lr_steps, cfg.lr_starts, cfg.lr_ends, num_steps)
    get_logger()('==> Learning Rate Scheduler: {}'.format(scheduler))
    return scheduler


#test
if __name__ == '__main__':
    import torch
    import argparse

    parser_ = argparse.ArgumentParser('Scheduler Test')
    options(parser_)
    # parser_.print_help()
    cfg_ = parser_.parse_args([
        # '--lr-methods=fix,cos,step,exp',
        '--lr-methods=step, cos,  cos, cos,fix',
        '--lr-starts=  0.1,   1,  0.5, 0.1',
        '--lr-ends=      1, 0.01, 0.01, 0.01',
        '--lr-steps=    5, 40,  20, 10'
    ])
    print(cfg_)
    optimizer_ = torch.optim.SGD(torch.nn.Linear(10, 10).parameters(), lr=0.1)
    scheduler_ = make(optimizer_, cfg_, 10)
    scheduler_.draw(80, log_scale=True, lr=cfg_.lr)
    print('state_dict:', scheduler_.state_dict())
