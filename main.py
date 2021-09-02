import time
from collections import OrderedDict

import torch
import torch.nn as nn
import extension as ext
import apex
import pdb
import os

from ExpSDN.datasets import charades_sta_i3d, charades_sta_vgg, tacos_i3d, tacos_c3d,activitynet_c3d 
from ExpSDN import model, loss


from extension.utils_tlg import eval_utils

class MyProjects(ext.Framework):
    '''
    using framework to design a dl project.
    '''
    def __init__(self) -> None:
        self.train_transforms = None
        self.train_dataset = None
        self.train_loader = None
        self.eval_transforms = None
        self.eval_dataset = None
        self.eval_loader = None
        self.test_transforms = None
        self.test_dataset = None
        self.test_loader = None
        super().__init__()
        self.eval_recorder = OrderedDict()
        self.best_score = 0
        self.best_epoch = -1
        self.init("best_score")
        self.init("best_epoch")
        return
    
    def step_3_transform_dataset_dataloader(self, *args, **kwargs):

        dsets, loader = charades_sta_vgg.create_loaders(self.cfg.loader_config)
        self.train_dataset = dsets["train"]
        self.train_loader = loader["train"]
        self.eval_dataset = dsets["eval"]
        self.eval_loader = loader["eval"]
        self.test_dataset = dsets["test"]
        self.test_loader = loader["test"]
   
    def step_4_model(self, *args, **kwargs):
        # self.model_name = self.cfg.arch + "_" + self.cfg.backbone + "_" + self.cfg.head
        self.model_name = self.cfg.model["arch"]
        self.cfg.common_model_config = self.set_common_model_cfg() ##normalization, quantilization, norm_weights
        self.logger("==> Model Config [{}]:\n{}".format(self.model_name, self.cfg.common_model_config))
        # model.dataset_specific_config_update(self.cfg.model, self.train_dataset)
        # pdb.set_trace()
        self.model = model.make(self.cfg.model)

        self.criterion = loss.make(self.cfg.model)
        self.load_model() ##check if it is using load
        self.init("model")
        self.init("criterion")
        self.logger("==> Model [{}]:\n{}".format(self.model_name, self.model))
        self.logger("==> Criterion: {}".format(self.criterion))
        self.to_cuda()   
    
    def step_7_others(self, *args, **kwargs):
        self.enable_fp16_training(with_criterion=True)
        self.weight_init()
    
    def weight_init(self):
        pass

    def prepare_batch(self, batch):
        gt_list = ["start", "end", "localiz", "videoFeat_lengths", "factor", "fps", "time_start", "time_end", "tokens_lengths", "duration", "video"]

        both_list = ["videoFeat_lengths", "tokens_lengths"]

        net_inps, gts = {}, {}
        for k in batch.keys():
            if ext.utils_tlg.net_utils.istensor(batch[k]):
                item = batch[k].to(self.device)
            else:
                item = batch[k]
            if k in gt_list: gts[k] = item
            else: 
                net_inps[k] = item
            if k in both_list: net_inps[k] = item

        return net_inps, gts

    def run(self):
        '''
        epoch management
        entry of projects
        '''
        if self.cfg.test:
            self.evaluator = eval_utils.get_evaluator(self.cfg.loader_config["test"]["dataset"])
            self.timer = ext.TimeMeter(0, 1, len(self.test_loader))
            self.loss_meter = ext.DictMeter()
            self.logger.set_file(self.cfg.test_log, append =False)
            self.cfg.resume = False
            assert self.cfg.load
            self.test()
            return
        ext.distributed.synchronize()
        self.evaluator = eval_utils.get_evaluator(self.cfg.loader_config["eval"]["dataset"])
        self.timer = ext.TimeMeter(self.start_epoch, self.cfg.epochs, len(self.train_loader))
        self.eval_timer = ext.TimeMeter(0, 1, len(self.eval_loader))
        self.loss_meter = ext.DictMeter()
        
        self.enable_vis(env_name="simple_vis")
        self.logger.set_file(self.cfg.train_log, append =False)
        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        # self.timer.start()
        self.logger("======================  Begin {} ========================".format(now_date.replace('_', ' ')))
    
        self.eval(self.start_epoch-1)
        
        for epoch in range(self.start_epoch, self.cfg.epochs):
            if ext.distributed.get_world_size() > 1:
                self.train_loader.sampler.set_epoch(epoch)
            #training model for one epoch
            self.train(epoch)

            #eval current model per cfg.eval_epoch
            if self.eval_loader != None and (epoch+1)%self.cfg.eval_epoch == 0 :
                self.eval(epoch)
            
            self.start_epoch += 1
            self.save_checkpoint(self.model_name+'_'+self.cfg.checkpoint_save)
        
        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        self.logger("======================  End {} ========================".format(now_date.replace('_', ' ')))
        self.logger.set_file(self.cfg.test_log, append =False)
        self.cfg.resume = False
        # self.cfg.load = self.model_name+"_I3D_best.pth"
        self.cfg.load = os.path.join(self.output, self.model_name+"_"+self.cfg.best_model_save)
        self.load_model()
        self.test()
        ext.distributed.synchronize()
        
        if self.vis != None:
            self.vis.__del__()
        return

    def train(self,epoch):
        '''
        running a training cycle for one epoch
        '''

        fmt_str = 'Train [{:%dd}/{}][{:%dd}/{}]: {}, lr: {:.1e} , time: {} <- {}, avg: {}' % (
            len(str(self.cfg.epochs)), len(str(len(self.train_loader))))
        self.set_train()
        self.timer.start()
        self.loss_meter.reset()
        for step, data_batch in enumerate(self.train_loader):
            
            net_inps, gts = self.prepare_batch(data_batch)
            # pdb.set_trace()
            net_out = self.model(net_inps)
            for k, v in net_out.items():
                if torch.any(torch.isnan(v)):
                    pdb.set_trace()

            self.lr_scheduler.step(epoch, step)
            current_lr = self.lr_scheduler.get_lr(self.cfg.lr, epoch, step)
            loss = self.criterion(output = net_out, target = gts)
            self.execute_backward(loss["total_loss"])

            for k, v in loss.items():
                if torch.any(torch.isnan(v)):
                    pdb.set_trace()
                loss[k] = ext.distributed.reduce_tensor(v)

            self.global_step += 1
            self.timer.step()
            self.loss_meter.update(loss)
            del net_inps, gts, loss

            avg_loss_meter = self.loss_meter.get_average()
            if self.vis != None:
                idx = self.global_step
                for k, v in avg_loss_meter.items():
                    self.vis.add_scalar(k, v, global_step=idx)
                self.vis.add_scalar('learning rate', current_lr, global_step = idx)
            tof = (step+1) % self.cfg.print_frequency == 0 or (step+1) == len(self.train_loader)
            self.logger(fmt_str.format(
                epoch + 1, self.cfg.epochs, step+1, len(self.train_loader),
                self.loss_meter.average,
                current_lr,
                self.timer.sum,
                self.timer.expect,
                self.timer.avg_str
                ), end = '\r\n'[tof], to_file = tof)
        
    @torch.no_grad()
    def eval(self, epoch):
        self.set_eval()
        self.loss_meter.reset()
        self.eval_timer.reset()
        for key in self.evaluator.metrics:
            self.eval_recorder[key] = torch.tensor(0).type(torch.FloatTensor).cuda(self.device)
        self.eval_recorder['mIoU'] = torch.tensor(0).type(torch.FloatTensor).cuda(self.device)
        fmt_str = 'Eval [{}][{}/{}]: {}, time: {} <- {}, avg: {}'

        self.eval_timer.start()
        for step, data_batch in enumerate(self.eval_loader):
            net_inps, gts = self.prepare_batch(data_batch)
            net_outs = self.model(net_inps)
            loss = self.criterion(output = net_outs, target = gts)
            

            for k, v in loss.items():
                if torch.any(torch.isnan(v)):
                    pdb.set_trace()
                loss[k] = ext.distributed.reduce_tensor(v)

            self.loss_meter.update(loss)
            
            # fetch data
            start = net_outs["pred_start"].detach()
            end = net_outs["pred_end"].detach()

            startings = torch.argmax(start, dim = -1)
            endings   = torch.argmax(end, dim = -1)

            
            if type(self.eval_dataset) == activitynet_c3d.ACNET_C3D:
                startings = ((startings+1) / gts["videoFeat_lengths"]) * gts["duration"]
                endings = ((endings+1) / gts["videoFeat_lengths"]) * gts["duration"]
                vid_d = gts["duration"]
            else:
                startings = gts["factor"] * startings / gts["fps"]
                endings = gts["factor"] * (endings+1) / gts["fps"]
                vid_d = gts["factor"] * gts["videoFeat_lengths"] / gts["fps"]


            gt_ts = torch.stack((gts["time_start"], gts["time_end"]), dim=-1)
            pred = torch.stack((startings, endings), dim=-1)

            # prepare results for evaluation

            results = {"predictions": pred, "gts": gt_ts, "durations":vid_d}
            
            # compute performances
            self.evaluator.set_duration(results["durations"])
            rank1, rank5, miou = self.evaluator.eval(results["predictions"], results["gts"], rand_show = False)            
            
            for k,v  in rank1.items():
                self.eval_recorder[k] += torch.tensor(v).cuda(self.device).type(torch.FloatTensor)
            if isinstance(miou, torch.Tensor):
                self.eval_recorder["mIoU"] += (miou).clone().detach().type(torch.FloatTensor)
            else:
                self.eval_recorder["mIoU"] += torch.tensor(miou).cuda(self.device).type(torch.FloatTensor)
            del net_inps, gts, loss
            self.eval_timer.step()
            tof = (step+1) % self.cfg.print_frequency == 0 or (step+1) == len(self.eval_loader)
            self.logger(fmt_str.format(
                epoch+1, step+1, len(self.eval_loader),
                self.loss_meter.average,
                self.eval_timer.sum,
                self.eval_timer.expect,
                self.eval_timer.avg_str
                ), end = '\r\n'[tof], to_file = tof)
                
        # for distributed muti gpus training
        for k, v in self.eval_recorder.items():
            self.eval_recorder[k] = ext.distributed.reduce_tensor(v, return_mean=False)

        if not ext.is_main_process():
            ext.distributed.synchronize()
            return 0
        
        for k,v  in self.eval_recorder.items():
            self.eval_recorder[k] /= len(self.eval_dataset)

        score = self.eval_recorder[self.evaluator.get_metric()]
        eval_str = f'epoch {epoch+1}: '
        for k, v in self.eval_recorder.items():
            eval_str += ' {}:{:.4f} '.format(k, v)
        self.logger(eval_str)
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch+1
            self.logger(f'Best score: {self.best_score:.4%} at epoch {self.best_epoch}')
            self.save_model(self.model_name+'_'+self.cfg.best_model_save)
        ext.distributed.synchronize()
        return score

    @torch.no_grad()            
    def test(self):
        self.set_eval()
        self.timer.start()
        self.loss_meter.reset()
        test_str = 'Test {}/{}, time: {} <- {}, avg: {}'
        
        for key in self.evaluator.metrics:
            self.eval_recorder[key] = torch.tensor(0).type(torch.FloatTensor).cuda(self.device)
        self.eval_recorder['mIoU'] = torch.tensor(0).type(torch.FloatTensor).cuda(self.device)
        self.timer.start()
        for step, data_batch in enumerate(self.test_loader):
            net_inps, gts = self.prepare_batch(data_batch)
            # pdb.set_trace()
            # print(net_inps["raw_tokens"])
            net_outs = self.model(net_inps)
            loss = self.criterion(output = net_outs, target = gts)

            for k, v in loss.items():
                loss[k] = ext.distributed.reduce_tensor(v)

            self.loss_meter.update(loss)

            # fetch data
            start = net_outs["pred_start"].detach()
            end = net_outs["pred_end"].detach()

            startings = torch.argmax(start, dim = -1)
            endings   = torch.argmax(end, dim = -1)
            

            if type(self.eval_dataset) == activitynet_c3d.ACNET_C3D:
                startings = ((startings+1) / gts["videoFeat_lengths"]) * gts["duration"]
                endings = ((endings+1) / gts["videoFeat_lengths"]) * gts["duration"]
                vid_d = gts["duration"]
            else:
                startings = gts["factor"] * startings / gts["fps"]
                endings = gts["factor"] * (endings+1) / gts["fps"]
                vid_d = gts["factor"] * gts["videoFeat_lengths"] / gts["fps"]

            gt_ts = torch.stack((gts["time_start"], gts["time_end"]), dim=-1)
            pred = torch.stack((startings, endings), dim=-1)

            results = {"predictions": pred, "gts": gt_ts, "durations": vid_d}


            self.evaluator.set_duration(results["durations"])
            rank1, rank5, miou = self.evaluator.eval(results["predictions"], results["gts"], rand_show = False)            
            
            for k,v  in rank1.items():
                self.eval_recorder[k] += torch.tensor(v).cuda(self.device).type(torch.FloatTensor)
            if isinstance(miou, torch.Tensor):
                self.eval_recorder["mIoU"] += (miou).clone().detach().type(torch.FloatTensor)
            else:
                self.eval_recorder["mIoU"] += torch.tensor(miou).cuda(self.device).type(torch.FloatTensor)

            del net_inps, gts, loss
            self.timer.step()
            tof = (step+1) % self.cfg.print_frequency == 0 or (step+1) == len(self.test_loader)
            self.logger(test_str.format(
                step+1, len(self.test_loader),
                self.timer.sum,
                self.timer.expect,
                self.timer.avg_str
                ), end = '\r\n'[tof], to_file = tof)

        # for distributed muti gpus training
        for k, v in self.eval_recorder.items():
            self.eval_recorder[k] = ext.distributed.reduce_tensor(v, return_mean=False)

        if not ext.is_main_process():
            ext.distributed.synchronize()
            return 0

        for k,v  in self.eval_recorder.items():
            self.eval_recorder[k] /= len(self.test_dataset)
        
        test_str = 'test {}: '.format(self.cfg.load)
        for k, v in self.eval_recorder.items():
            test_str += ' {}:{:6.4f} '.format(k , v)
        self.logger(test_str)

        ext.distributed.synchronize()
        return 

    
            


if __name__ == "__main__":
    mnist_project = MyProjects()
    mnist_project.run()
