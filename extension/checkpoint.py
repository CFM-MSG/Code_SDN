from extension.config import get_parser
from . import utils


def options(parser=None):
    if parser is None:
        parser = get_parser()
    group = parser.add_argument_group("Save Options")
    group.add_argument("--resume", default=None, metavar="P", type=utils.path,
                       help="path to the checkpoint needed resume")
    group.add_argument("--load", default=None, metavar="P", type=utils.path,
                       help="The path to (pre-)trained model.")
    group.add_argument("--load-no-strict", default=False, action="store_true",
                       help="The keys of loaded model may not exactly match the model's. (May usefully for finetune)")
    return


_checkpoint = {}


def set_checkpoint(ck):
    global _checkpoint
    _checkpoint = ck


def get_from_checkpoint(name):
    if name in _checkpoint:
        return _checkpoint.pop(name)
    else:
        return None