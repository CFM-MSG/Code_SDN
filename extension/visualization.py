import argparse
from collections import defaultdict
from os import mkdir
import os

import numpy as np
import torch
import datetime

from extension.config import get_parser
from extension.distributed import is_main_process
from extension.utils import str2bool

# opts.title : 图标题
# opts.width : 图宽
# opts.height : 图高
# opts.showlegend : 显示图例 (true or false)
# opts.xtype : x轴的类型 ('linear' or 'log')
# opts.xlabel : x轴的标签
# opts.xtick : 显示x轴上的刻度 (boolean)
# opts.xtickmin : 指定x轴上的第一个刻度 (number)
# opts.xtickmax : 指定x轴上的最后一个刻度 (number)
# opts.xtickvals : x轴上刻度的位置(table of numbers)
# opts.xticklabels : 在x轴上标记标签 (table of strings)
# opts.xtickstep : x轴上刻度之间的距离 (number)
# opts.xtickfont :x轴标签的字体 (dict of font information)
# opts.ytype : type of y-axis ('linear' or 'log')
# opts.ylabel : label of y-axis
# opts.ytick : show ticks on y-axis (boolean)
# opts.ytickmin : first tick on y-axis (number)
# opts.ytickmax : last tick on y-axis (number)
# opts.ytickvals : locations of ticks on y-axis (table of numbers)
# opts.yticklabels : ticks labels on y-axis (table of strings)
# opts.ytickstep : distances between ticks on y-axis (number)
# opts.ytickfont : font for y-axis labels (dict of font information)
# opts.marginleft : 左边框 (in pixels)
# opts.marginright :右边框 (in pixels)
# opts.margintop : 上边框 (in pixels)
# opts.marginbottom: 下边框 (in pixels)

class Visualization:
    def __init__(self, env_name, method='visdom', enable=False, port=6006, log_path = "./vis/"):
        self.vis = enable and is_main_process()
        self.port = port
        self.env_name = env_name
        self.method = method
        self.log_path = log_path
        self._data = {}
        if self.vis:
            if self.method == 'visdom':
                try:
                    import visdom
                    self.viz = visdom.Visdom(env=env_name, port=self.port)
                except ImportError:
                    self.vis = False
                    print("\033[031mYou do not install visdom!!!!\033[0m")
            else:
                if not os.path.exists(self.log_path):
                    mkdir(self.log_path)
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.writer = SummaryWriter(os.path.join(log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
                except ImportError:
                    self.vis = False
                    print("\033[031mYou do not install tensorboard!!!!\033[0m")

    def state_dict(self):
        return self._data

    def load_state_dict(self, state_dict):
        if not self.vis or self.method != 'visdom':
            return
        self._data = state_dict

        for tag, values in self._data.items():
            opts = dict(
                title=tag,
                # legend=self.windows[label],
                showlegend=False,
                # webgl=False,
                # layoutopts={"plotly": {"legend": {"x": 0.05, "y": 1}}},
                # marginleft=0, marginright=0, margintop=10, marginbottom=0,
            )
            update = None
            for k in values['x'].keys():
                self.viz.line(np.array(values['y'][k]), np.array(values['x'][k]),
                              update=update, win=tag, name=k, opts=opts)
                update = 'append'

    def add_scalar(self, tag, value_or_dict=None, global_step=None, **kwargs):
        if not self.vis:
            return
        if value_or_dict is None:
            value_or_dict = kwargs
        elif isinstance(value_or_dict, dict):
            value_or_dict.update(kwargs)
        else:
            assert len(kwargs) == 0

        if self.method == 'tensorboard':
            if isinstance(value_or_dict, dict):
                self.writer.add_scalars(tag, value_or_dict, global_step)
            else:
                self.writer.add_scalar(tag, value_or_dict, global_step)
            return
        ### Visdom ###
        if tag not in self._data:
            self._data[tag] = {
                'x': defaultdict(list),
                'y': defaultdict(list),
            }
            update = None
        else:
            update = 'append'
        opts = dict(
            title=tag,
            # legend=self.windows[label],
            showlegend=False,
            # webgl=False,
            # layoutopts={"plotly": {"legend": {"x": 0.05, "y": 1}}},
            # marginleft=0, marginright=0, margintop=10, marginbottom=0,
        )
        if not isinstance(value_or_dict, dict):
            value_or_dict = {tag: value_or_dict}
        for k, v in value_or_dict.items():
            if isinstance(v, torch.Tensor):
                assert v.numel() == 1
                y = v.item()
            else:
                y = v
            if global_step is None:
                x = len(self._data[tag]['y'][k])
            else:
                x = global_step
            self._data[tag]['x'][k].append(x)
            self._data[tag]['y'][k].append(y)
            self.viz.line(np.array([y]), np.array([x]), update=update, win=tag, name=k, opts=opts)
            update = 'append'

    def clear(self, win=None):
        if not self.vis or self.method != 'visdom':
            return
        if win is None:
            import visdom

            self.viz.delete_env(self.env_name)
            self.viz = visdom.Visdom(env=self.env_name, port=self.port)
            self._data.clear()
        elif win in self._data:
            self._data.pop(win)

    def add_images(self, images, title="images", win="images", nrow=8):
        if self.vis:
            if self.method == 'visdom':
                self.viz.images(images, win=win, nrow=nrow, opts={"title": title})
            else:
                raise NotImplemented

    def __del__(self):
        if self.vis:
            if self.method == 'visdom':
                self.viz.save([self.env_name])
                # self.viz.close()
            else:
                self.writer.close()

def options(parser=None):
    if parser is None:
        parser = get_parser()
    group = parser.add_argument_group("Visualization Options")
    group.add_argument("--vis", default=False, const=True, nargs='?', type=str2bool,
                       help="Is the visualization training process?")
    group.add_argument("--vis-port", default=6006, type=int, metavar='N', help="The visualization port (default 6006)")
    group.add_argument("--vis-method", default='tensorboard', metavar='C', choices=['visdom', 'tensorboard'],
                       help="The method to visualization (visdom/tensorboard)")
    return group

def make(cfg: argparse.Namespace, env_name: str, enable=True) -> Visualization:
    return Visualization(env_name, cfg.vis_method, cfg.vis and enable, cfg.vis_port)

