
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d
from .utils import *

class ModelSelector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args     = args
        self.bm       = self.args.TRAIN.BM
        self.folder   = self.args.MODEL.SAVEDIR
        self.wtfolder = self.args.TRAIN.WTFOLDER
        self.ssltype  = self.args.TRAIN.SSL_TYPE
        self.wts      = self.args.TRAIN.WTS

        self.infeat = 2048 if self.args.TRAIN.BM == 'resnet50' else 512
        self.num_class = self.args.DATA.NUM_TRIPLET

        self.generate_model()

        # linear classifier from resnet features to num triplet classes
        self.fc = nn.Linear(self.infeat, self.num_class, bias=False)

    def generate_model(self):
        if self.wts == 'imagenet':
            if self.bm == 'resnet18':
                self.backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
            elif self.bm == 'resnet50':
                self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-1])

                # freeze backbone if needed
                if self.args.TRAIN.LINEARPROB:
                    for k, v in self.backbone.named_parameters():
                        v.requires_grad = False
                    self.backbone = freeze_batch_norm_2d(self.backbone)
        elif self.wts == 'ssl':
            self.backbone = resnet50(pretrained=True)
            ssl_wt = os.path.join(self.folder, self.wtfolder, f'model_final_checkpoint_{self.ssltype}_surg.torch')
            
            # load checkpoints
            if self.args.exp_mode == 'train':
                assert os.path.exists(ssl_wt)
                logging.info(f"SSL checkpoint used >>> {ssl_wt}")
                load_ckpt(self.backbone, ssl_wt)

            # freeze backbone if needed
            if self.args.TRAIN.LINEARPROB:
                for k, v in self.backbone.named_parameters():
                    v.requires_grad = False
                self.backbone = freeze_batch_norm_2d(self.backbone)

            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError("incorrect weights specified!")

    def forward(self, x):
        out = self.backbone(x).squeeze(-1).squeeze(-1)
        logit = self.fc(out)

        return logit

def get_model(args):
    model = ModelSelector(args)

    # load model to cuda if available
    if torch.cuda.is_available():
        if isinstance(args.TRAIN.GPUS, int):
            args.TRAIN.GPUS = [args.TRAIN.GPUS]
        model = nn.DataParallel(model, device_ids=args.TRAIN.GPUS)
        model = model.cuda()

    return model

if __name__ == "__main__":
    pass


