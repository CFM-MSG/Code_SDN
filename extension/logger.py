import os
import shutil
import time
import warnings

from extension.config import get_parser
from .distributed import is_main_process

_color = {
    'black': '0',  # 黑色字
    'red': '1',  # 红色字
    'green': '2',  # 绿色字
    'yellow': '3',  # 黄色字
    'blue': '4',  # 蓝色字
    'purple': '5',  # 紫色字
    'sky_blue': '6',  # 天蓝字
    'white': '7',  # 白色字
}


class _Logger:
    def __init__(self, to_stdout=True, to_file=True):
        self.to_stdout = to_stdout
        self.to_file = to_file
        self.file = None
        self.filename = None
        self._buffer = ""  # when to_file=True and file is not set, write content to buffer

    def __del__(self):
        if self.file:
            if self._buffer:
                self.file.write(self._buffer)
                self._buffer = ""
            self.file.close()
        if self._buffer and self.to_file and not self.filename:
            warnings.warn(
                "The log is not write to file. Please use <logger>.set_file(*paths, append=False) to set log file.")

    def set_file(self, *paths, append=False):
        self.filename = os.path.join(*paths)
        if self.to_file:
            self.file = open(self.filename, "a" if append else "w")
            self.__call__("==> Save log on {}".format(os.path.abspath(self.filename)))

    def __call__(self, *msg, sep=' ', end="\n", to_stdout=True, to_file=True, fg_color=None, bg_color=None):
        """
        sep:   string inserted between values, default a space.
        end:   string appended after the last value, default a newline.
        """
        if to_file and self.to_file:
            self._buffer += sep.join([str(m) for m in msg]) + end
            if self.file is not None:
                self.file.write(self._buffer)
                self._buffer = ""
                self.file.flush()
        if to_stdout and self.to_stdout:
            if fg_color is not None or bg_color is not None:
                color_s = '\033['
                if fg_color is not None:
                    color_s += '3' + _color[fg_color]
                    if bg_color is not None:
                        color_s += ';'
                if bg_color is not None:
                    color_s += '4' + _color[bg_color]
                color_s += 'm'
                msg = [color_s] + list(msg) + ['\033[0m']
            print(*msg, sep=sep, end=end)

    def INFO(self, *msg, sep=' ', end="\n", to_stdout=True, to_file=True):
        self(*msg, sep=sep, end=end, to_stdout=to_stdout, to_file=to_file)

    def DEBUG(self, *msg, sep=' ', end="\n", to_stdout=True, to_file=True):
        self(*msg, sep=sep, end=end, to_stdout=to_stdout, to_file=to_file, fg_color='black', bg_color='white')

    def WARN(self, *msg, sep=' ', end="\n", to_stdout=True, to_file=True):
        self(*msg, sep=sep, end=end, to_stdout=to_stdout, to_file=to_file, fg_color='green', bg_color='white')

    def ERROR(self, *msg, sep=' ', end="\n", to_stdout=True, to_file=True):
        self(*msg, sep=sep, end=end, to_stdout=to_stdout, to_file=to_file, fg_color='red', bg_color='white')

    def NOTE(self, *msg, sep=' ', end="\n", to_stdout=True, to_file=True):
        self(*msg, sep=sep, end=end, to_stdout=to_stdout, to_file=to_file, fg_color='blue', bg_color='white')

    def copy(self, filename):
        if self.file is not None:
            shutil.copy(self.filename, filename)
            print("Copy log file to {}".format(filename))


logger = _Logger(True, False)


def options(parser=None):
    if parser is None:
        parser = get_parser()
    group = parser.add_argument_group("Logger Options")
    group.add_argument('--no-log', action='store_true', help='Do not log to file')
    # group.add_argument("--log-suffix", metavar="S", default="", help="the suffix of log path.")
    group.add_argument("--print-frequency", metavar="N", default=100, type=int, help="print frequency. (default: 100)")
    return group
def make(to_stdout=True, to_file=True):
    global logger
    if is_main_process():
        logger = _Logger(to_stdout, to_file)
        logger("==> Start log at {}.".format(time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))))
    else:
        logger = _Logger(False, False)
    return logger


def get_logger():
    return logger
