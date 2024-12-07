from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from configs.__base__ import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_lgc):

    def __init__(self):
        # super(cfg, self).__init__()
        cfg_common.__init__(self)
        cfg_dataset_default.__init__(self)
        cfg_model_lgc.__init__(self)

        self.fvcore_b = 1
        self.fvcore_c = 3
        self.seed = 42
        self.size = 256
        self.epoch_full = 200
        self.warmup_epochs = 0
        self.test_start_epoch = self.epoch_full
        self.test_per_epoch = self.epoch_full // 10
        self.batch_train = 16  # official 16
        self.batch_test_per = 16
        self.lr = 0.001 * self.batch_train / 16
        self.weight_decay = 0.05
        self.metrics = [
            'mAUROC_sp_max',
            'mAUPRO_px',
            'mAUROC_px'
        ]
        self.use_adeval = True
        self.lambda_1 = 1
        self.lambda_2 = 1
        self.vis = False
        

        # ==> data
        # self.data.type = 'UnifiedAD'
        self.data.type = 'DefaultAD'
        self.data.root = '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/hjj/hjj/data/MVTec_AD'
        # self.data.root = r'/mnt/lpai-dione/ssai/cvg/team/share_datasets/MVTec_AD'
        # self.data.root = r'/media/huzhiguang/Work/WorkSpace/Project/Python/Dataset/General/AD/MVTec AD'
        # self.data.root = [
        #     '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/hjj/hjj/data/BTech_Dataset_Transformed',
        #     '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/hjj/hjj/data/MVTec_AD',
        #     '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/hjj/hjj/data/VisA',
        #     # '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/hjj/hjj/data/realiad_512'
        # ]
        self.data.anomaly_source_path = '/mnt/pfs-mc0p4k/cvg/team/hjj/code/diffusionad/data/dtd/images'
        # self.data.anomaly_source_path = '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/hjj/hjj/data/dtd/images'
        self.data.meta = 'meta.json'
        self.data.resize_shape = [self.size, self.size]
        self.data.cls_names = []
        self.data.use_sample = True

        self.data.train_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ]
        self.data.test_transforms = self.data.train_transforms
        self.data.target_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
        ]
        self.data.aug_transforms = [
            dict(type='RandomResizedCrop', size=(self.size, self.size), scale=(0.8, 1.0)),
            dict(type='RandomHorizontalFlip', p=0.5),
            dict(type='RandomVerticalFlip', p=0.5),
            dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            dict(type='RandomGrayscale', p=0.2),
            dict(type='GaussianBlur', kernel_size=23, sigma=(0.1, 2.0)),
            dict(type='RandomRotation', degrees=15),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ]

        # ==> modal
        # checkpoint_path = '/media/huzhiguang/Work/WorkSpace/Project/Python/HJJ/code/ader/runs/RDADTrainer_configs_rdad_rd_256_100e_20241113-173245/net_40.pth'
        checkpoint_path = ''
        self.model_t = Namespace()
        self.model_t.name = 'timm_wide_resnet50_2'
        self.model_t.kwargs = dict(pretrained=True, checkpoint_path='',
                                   strict=False, features_only=True, out_indices=[1, 2, 3])
        self.model_s = Namespace()
        self.model_s.name = 'de_wide_resnet50_2'
        self.model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)
        self.model = Namespace()
        self.model.name = 'lgc'
        self.model.kwargs = dict(pretrained=False, checkpoint_path=checkpoint_path, strict=True, model_t=self.model_t,
                                 model_s=self.model_s)

        # ==> evaluator
        self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=None, max_step_aupro=100,
                                     use_adeval=self.use_adeval)

        # ==> optimizer
        self.optim.proj_opt = Namespace()
        self.optim.distill_opt = Namespace()
        self.optim.lr = self.lr
        self.optim.proj_opt.kwargs = dict(name='adam', betas=(0.5, 0.999))
        self.optim.distill_opt.kwargs = dict(name='adam', betas=(0.5, 0.999))

        # ==> trainer
        self.trainer.name = 'LGCTrainer'
        self.trainer.logdir_sub = ''
        self.trainer.resume_dir = ''
        self.trainer.epoch_full = self.epoch_full
        self.trainer.scheduler_kwargs = dict(
            name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2,
            warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs,
            cooldown_epochs=0, use_iters=True,
            patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8),
            cycle_decay=0.1, decay_rate=0.1)
        self.trainer.mixup_kwargs = dict(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0,
                                         switch_prob=0.5, mode='batch', correct_lam=True, label_smoothing=0.1)
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch

        self.trainer.data.batch_size = self.batch_train
        self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

        # ==> loss
        self.loss.loss_terms = [
            dict(type='CosLoss', name='cos', avg=False, lam=1.0),
            dict(type='DenseLoss', name='dense', lam=1.0, temperature=0.05),
            dict(type='SCLLoss', name='scl', lam=1.0, temperature=0.05),
        ]

        # ==> logging
        self.logging.log_terms_train = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='data_t', fmt=':>5.3f'),
            dict(name='optim_t', fmt=':>5.3f'),
            dict(name='lr', fmt=':>7.6f'),
            dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
            dict(name='glb', suffixes=[''], fmt=':>5.3f', add_name='avg'),
            dict(name='dense', suffixes=[''], fmt=':>5.3f', add_name='avg'),
            dict(name='acl', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]
        self.logging.log_terms_test = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
            dict(name='glb', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]
