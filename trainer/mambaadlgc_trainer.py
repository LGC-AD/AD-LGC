import os
import copy
import glob
import shutil
import datetime
import time

import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp, read_data, save_data


@TRAINER.register_module
class MAMBAADLGCTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(MAMBAADLGCTrainer, self).__init__(cfg)
        self.optim.proj_opt = get_optim(cfg.optim.proj_opt.kwargs, self.net.proj_layer, lr=cfg.optim.lr)
        proj_layer = self.net.proj_layer
        self.net.proj_layer = None
        self.optim.distill_opt = get_optim(cfg.optim.distill_opt.kwargs, self.net, lr=cfg.optim.lr * 5)
        self.net.proj_layer = proj_layer

    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.bs = self.imgs.shape[0]
        self.aug_imgs = inputs.get('aug_img', None)
        self.labels = inputs['label'].cuda()
        if self.aug_imgs is not None:
            self.aug_imgs = self.aug_imgs.cuda()


    def forward(self):
        self.feats_t, self.feats_s, self.feats_t_k, self.feats_t_q_grid, self.feats_t_k_grid, self.glb_feats = self.net(
            self.imgs, self.aug_imgs)

    def backward_term(self, loss_term, optim):
        optim.proj_opt.zero_grad()
        optim.distill_opt.zero_grad()
        if self.loss_scaler:
            self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(),
                             create_graph=self.cfg.loss.create_graph)
        else:
            loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)

            optim.proj_opt.step()
            optim.distill_opt.step()

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
            loss_glb = self.loss_terms['scl'](self.glb_feats, self.labels)
            loss_den = self.loss_terms['dense'](self.feats_t, self.feats_t_k, self.feats_t_q_grid, self.feats_t_k_grid,
                                                self.labels)
            loss = loss_mse + loss_glb + loss_den
        self.backward_term(loss, self.optim)

        update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(), 1,
                        self.master)
        update_log_term(self.log_terms.get('glb'), reduce_tensor(loss_glb, self.world_size).clone().detach().item(), 1,
                        self.master)
        update_log_term(self.log_terms.get('dense'), reduce_tensor(loss_den, self.world_size).clone().detach().item(),
                        1, self.master)

    # def train(self):
    #     self.reset(isTrain=True)
    #     self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
    #     train_length = self.cfg.data.train_size
    #     train_loader = iter(self.train_loader)
    #     while self.epoch < self.epoch_full and self.iter < self.iter_full:
    #         self.scheduler_step(self.iter)
    #         # ---------- data ----------
    #         t1 = get_timepc()
    #         self.iter += 1
    #         train_data = next(train_loader)
    #         self.set_input(train_data)
    #         t2 = get_timepc()
    #         update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
    #         # ---------- optimization ----------
    #         self.optimize_parameters()
    #         t3 = get_timepc()
    #         update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
    #         update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
    #         self.cfg.total_time = get_timepc() - self.cfg.task_start_time
    #         self.save_checkpoint()
    #         break
    #         # ---------- log ----------
    #         if self.master:
    #             if self.iter % self.cfg.logging.train_log_per == 0:
    #                 msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length,
    #                                                  self.iter_full / train_length), self.master, None)
    #                 log_msg(self.logger, msg)
    #                 if self.writer:
    #                     for k, v in self.log_terms.items():
    #                         self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
    #                     self.writer.flush()
    #         if self.iter % self.cfg.logging.train_reset_log_per == 0:
    #             self.reset(isTrain=True)
    #         # ---------- update train_loader ----------
    #         if self.iter % train_length == 0:
    #             self.epoch += 1
    #             if self.cfg.dist and self.dist_BN != '':
    #                 distribute_bn(self.net, self.world_size, self.dist_BN)
    #             self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None
    #             if self.epoch >= self.cfg.trainer.test_start_epoch or self.epoch % self.cfg.trainer.test_per_epoch == 0:
    #                 self.test()
    #             else:
    #                 self.test_ghost()
    #             self.cfg.total_time = get_timepc() - self.cfg.task_start_time
    #             total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
    #             eta_time_str = str(
    #                 datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
    #             log_msg(self.logger,
    #                     f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
    #             self.save_checkpoint()
    #             self.reset(isTrain=True)
    #             self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
    #             train_loader = iter(self.train_loader)
    #     self._finish()

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)
        imgs_masks, anomaly_maps, cls_names, anomalys, sample_anomalys, sample_predicts = [], [], [], [], [], []
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            # if batch_idx == 10:
            # 	break
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()
            loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
            update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(),
                            1, self.master)
            # get anomaly maps
            anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s,
                                                            [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False,
                                                            amap_mode='add', gaussian_sigma=4)
            # self.imgs_mask[self.imgs_mask > 0.], self.imgs_mask[self.imgs_mask <= 0.] = 1, 0
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            if self.cfg.vis:
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.writer.logdir
                vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map,
                               self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
            save_path = os.path.join(self.cfg.logdir, 'results')
            save_data(save_path, self.cls_name, self.img_path, self.imgs_mask.cpu().numpy().astype(int), anomaly_map,
                      self.anomaly.cpu().numpy().astype(int))

            # imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            # anomaly_maps.append(anomaly_map)
            # cls_names.append(np.array(self.cls_name))
            # anomalys.append(self.anomaly.cpu().numpy().astype(int))
            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
            # ---------- log ----------
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
                    log_msg(self.logger, msg)
            # self.writer.add_embedding(torch.cat(glb_feats, dim=0), metadata=torch.cat(labels, dim=0), global_step=self.epoch)
            # merge results
            # if self.cfg.dist:
            #     results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
            #     torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
            #     if self.master:
            #         results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
            #         valid_results = False
            #         while not valid_results:
            #             results_files = glob.glob(f'{self.tmp_dir}/*.pth')
            #             if len(results_files) != self.cfg.world_size:
            #                 time.sleep(1)
            #             else:
            #                 idx_result = 0
            #                 while idx_result < self.cfg.world_size:
            #                     results_file = results_files[idx_result]
            #                     try:
            #                         result = torch.load(results_file)
            #                         for k, v in result.items():
            #                             results[k].extend(v)
            #                         idx_result += 1
            #                     except:
            #                         time.sleep(1)
            #                 valid_results = True
            # else:
            #     results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
        if self.master:
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                imgs_masks, anomaly_maps, anomalys, cls_names = read_data(save_path, cls_name)
                results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, anomalys=anomalys, cls_names=cls_names)

                results = {k: np.array(v) for k, v in results.items()}
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None
                # msg += f'\n{cls_name:<10}'
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center",
                                    stralign="center", )
            log_msg(self.logger, f'\n{msg}')
            shutil.rmtree(save_path)