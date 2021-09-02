
import os
import torch
import warnings
import abc
from collections import OrderedDict

from .checkpoint import get_from_checkpoint
import extension as ext

from torch.optim.optimizer import Optimizer

try:
    import apex

    amp = apex.amp
except ImportError:
    apex = None
    amp = None


def _strip_prefix_if_present(state_dict, prefix):
    '''
    strip module parameters by names
    use this func to extract module from total dict
    '''
    keys = sorted(state_dict.keys())
    #unless exists a key without prefix in the state_dict, it all consists of network modules.
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key[len(prefix):]] = value
    #get splited modules.
    return stripped_state_dict

def _add_prefix_if_not_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if any(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[prefix + key] = value
    return stripped_state_dict

class Framework:
    """A framework for pytorch model train, evaluation, test"""

    def __init__(self):
        self.description = self.__doc__
        self.model_name = ""
        self.model_cfg = ""
        self.output = ""
        self.device = None  # type: Optional[torch.device]
        self.model = None  # type: Optional[torch.nn.Module]
        self.criterion = None  # type: Optional[torch.nn.Module]
        self.optimizer = None  # type: Optional[torch.optim.SGD]
        self.lr_scheduler = None  # type: Optional[ext.scheduler.LRScheduler]
        # self.train_dataset = None  # type: Optional[Dataset]
        # self.eval_dataset = None  # type: Optional[Dataset]
        # self.test_dataset = None  # type: Optional[Dataset]
        # self.train_loader = None  # type: Optional[DataLoader]
        # self.eval_loader = None  # type: Optional[DataLoader]
        # self.test_loader = None  # type: Optional[DataLoader]
        self.cfg = None  # type:Optional[argparse.Namespace]
        self.logger = None  # type: Optional[ext.logger._Logger]
        self.vis = None  # type: Optional[ext.visualization.Visualization]
        self._save_for_checkpoint = []
        self.global_step = 0
        self.start_epoch = 0
        self.amp = amp

        self.step_1_config()
        self.step_2_environment()
        self.step_3_transform_dataset_dataloader()
        self.step_4_model()
        self.step_5_optimizer()
        self.step_6_lr()
        self.step_7_others()
        self.logger("==> Config:")
        for k, v in sorted(vars(self.cfg).items()):
            if isinstance(v, dict):
                self.logger("{}:".format(k))
                for i_k, i_v in v.items():
                    if isinstance(i_v, dict):
                        self.logger("  {}:".format(i_k))
                        for ii_k, ii_v in i_v.items():
                            self.logger("   {}: {}".format(ii_k, ii_v))
                    else:
                        self.logger("  {}: {}".format(i_k, i_v))
            else:
                self.logger("{}: {}".format(k, v))

    def step_1_config(self):
        parser = ext.get_parser(self.description)
        # config
        ext.config.options()
        # environment
        ext.distributed.options()
        ext.trainer.options()
        ext.logger.options()
        ext.visualization.options()
        # model
        ext.normalization.options()
        ext.norm_weight.options()
        ext.checkpoint.options()
        # solver
        ext.scheduler.options()
        ext.optimizer.options()
        # extra
        self.extra_config(parser)
        print("making cfg...")
        self.cfg = ext.config.make() #loading cfg from yaml and checkpoint(if used resume)
        # print(self.cfg)
        self.init('cfg')
        return 

    def step_2_environment(self, *args, **kwargs):
        self.output = self.cfg.output
        os.makedirs(self.output, exist_ok=True)
        self.device = ext.distributed.make(self.cfg)
        self.logger = ext.logger.make(to_file=not self.cfg.no_log)
        ext.trainer.make(self.cfg)
        self.extra_environment(*args, **kwargs)
        self.init('global_step')
        self.init('start_epoch')
        
        if self.cfg.fp16 and self.amp is None:
            self.cfg.fp16 = False
            self.logger.ERROR('Install apex from https://github.com/nvidia/apex for mixed-precision training.')
        return

    def extra_environment(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step_3_transform_dataset_dataloader(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step_4_model(self, *args, **kwargs):
        self.get_common_model_cfg()
        pass

    def step_5_optimizer(self, *args, **kwargs):
        self.optimizer = ext.optimizer.make(self.model, self.cfg)
        self.init("optimizer")
        return

    def step_6_lr(self, *args, **kwargs):
        self.lr_scheduler = ext.scheduler.make(self.optimizer, self.cfg, len(self.train_loader))
        self.init("lr_scheduler")
        return

    def step_7_others(self, *args, **kwargs):
        pass

    def extra_config(self, *args, **kwargs):
        pass

    def execute_backward(self, loss, optimizer: Optimizer = None, retain_graph = False):
        if optimizer is None:
            optimizer = self.optimizer
        optimizer.zero_grad()
        if self.cfg.fp16:
            with self.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph = retain_graph)
        else:
            loss.backward(retain_graph = retain_graph)
        if self.cfg.grad_clip > 0:
            if self.cfg.fp16:
                torch.nn.utils.clip_grad_norm_(self.amp.master_params(optimizer), self.cfg.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        # loss.backward(retain_graph = retain_graph)
        optimizer.step()

    def enable_vis(self, env_name):
        self.vis = ext.visualization.make(self.cfg, env_name, enable=True)
        self.init('vis')

    def enable_fp16_training(self, with_criterion=False):
        if not self.cfg.fp16:
            return
        if self.criterion and with_criterion:
            (self.model, self.criterion), self.optimizer = self.amp.initialize(
                [self.model, self.criterion], self.optimizer, opt_level='O1')
        else:
            self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer, opt_level='O1')
        if not self.cfg.distributed or self.cfg.test:
            return
        self.model = self._to_parallel(self.model)
        if self.criterion and with_criterion:
            self.criterion = self._to_parallel(self.criterion)

    def set_common_model_cfg(self):
        model_cfg = ext.normalization.make(self.cfg)
        model_cfg += ext.norm_weight.make(self.cfg)
        return model_cfg

    def init(self, name):
        """
        init configuration in this running framework work.
        if it exist in the checkpoint(resume), using existing data.
        if <name> is a module, which can "load_state_dict" and "state_dict", using them. (For model, optimizer, lr_scheduler)
        """
        self._save_for_checkpoint.append(name)
        #target is not in checkpoint
        value = get_from_checkpoint(name)
        if value is None:
            return
            
        #if it is a Module class, value is the state_dict
        if hasattr(getattr(self, name), "load_state_dict"): 
            obj = getattr(self, name)
            if isinstance(obj, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                saved = _add_prefix_if_not_present(value, "module.")
            else:
                saved = _strip_prefix_if_present(value, "module.")
            # saved = _strip_prefix_if_present(value, "module.")
            obj.load_state_dict(saved)
            self.logger.NOTE("==> Load state dict `{}` from checkpoint".format(name, value))
         #it is not a module but still exists in checkpoint, like optimizer or lr_scheduler, use them.
        else:
            setattr(self, name, value)
            self.logger.NOTE("==> Resume `{}`={}".format(name, value))
        return
        
    #this is for the whole framework obj(cfg+model)
    def save_checkpoint(self, filename="checkpoint.pth", **kwargs):
        if not ext.distributed.is_main_process() or self.cfg.test:
            return
        data = kwargs
        for name in self._save_for_checkpoint:
            if hasattr(getattr(self, name), "state_dict"):
                data[name] = _strip_prefix_if_present(getattr(self, name).state_dict(), "module.")
            else:
                data[name] = getattr(self, name)
        path = os.path.join(self.output, filename)
        self.logger("Save checkpoint to {}".format(path))
        torch.save(data, path)
        return

    #this is noly for networks models.
    def save_model(self, name="model.pth"):
        if not ext.distributed.is_main_process():
            return
        path = os.path.join(self.output, name)
        data = _strip_prefix_if_present(self.model.state_dict(), "module.")
        self.logger("Saving model to {}".format(path))
        torch.save(data, path)
        return
    
    def load_model(self):
        if not self.cfg.load or self.cfg.resume:
            return
        # if not ext.is_main_process():
        #     ext.distributed.synchronize()
        #     return 0
        self.logger.NOTE('==> Loading model from {}, strict: {}'.format(self.cfg.load, not self.cfg.load_no_strict))
        loaded_state_dict = torch.load(self.cfg.load, map_location=torch.device("cpu"))

        if 'model' in loaded_state_dict:
            warnings.warn('There are {} in saved model.'.format(list(loaded_state_dict.keys())))
            loaded_state_dict = loaded_state_dict['model']
        # for DataParallel or DistributedDataParallel
        loaded_state_dict = _strip_prefix_if_present(loaded_state_dict, "module.")
        if isinstance(self.model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel, apex.parallel.distributed.DistributedDataParallel)):
            loaded_state_dict = _add_prefix_if_not_present(loaded_state_dict, "module.")
        self.model.load_state_dict(loaded_state_dict, strict=not self.cfg.load_no_strict)
        ext.distributed.synchronize()
        return

    def to_cuda(self, with_criterion=False):
        assert self.model is not None and self.criterion is not None, "Please define model and criterion first"
        self.model.to(self.device)
        if self.criterion and with_criterion:  # Some criterion is in the model
            for cri in self.criterion:
                self.criterion.to(self.device)
        if not self.cfg.distributed or self.cfg.test:
            return
        self.model = self._to_parallel(self.model)
        if self.criterion and with_criterion:
            for cri in self.criterion:
                self.criterion = self._to_parallel(self.criterion)

    def _to_parallel(self, model):
        # use DistributedDataParallel instead of DataParallel
        # command: CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
        # local_rank: used to specify which GPU main proc is using
        if self.cfg.dist_apex and apex is not None:
            if self.cfg.sync_bn:
                model = apex.parallel.convert_syncbn_model(model)
                self.logger("==> convert to SyncBatchNorm.")
            model = apex.parallel.DistributedDataParallel(model)
            self.logger("==> use DistributedDataParallel by apex")
        else:
            if self.cfg.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                self.logger("==> convert to SyncBatchNorm.")
              
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.cfg.local_rank], output_device=self.cfg.local_rank,
                find_unused_parameters=True)
            self.logger("==> use DistributedDataParallel by torch")
        return model
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, device_ids=[self.cfg.local_rank],
        #     output_device=self.cfg.local_rank, 
        #     find_unused_parameters=True
        # )
        # self.logger("==> use DistributedDataParallel by torch")
        # return model

    def set_train(self):
        '''
        enable BN and Dropout
        '''
        self.model.train()

    def set_eval(self):
        '''
        disable BN and Dropout
        '''
        self.model.eval()



