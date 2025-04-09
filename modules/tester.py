import logging
import os
from abc import abstractmethod
import numpy as np
import time

import cv2
import torch

import pandas as pd

from .metrics_clinical import CheXbertMetrics

class BaseTester(object):
    def __init__(self, model, criterion_cls, metric_ftns, args, device):
        self.args = args
        self.model = model
        self.device = device

        self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)


        self.criterion_cls = criterion_cls
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

class Tester(BaseTester):
    def __init__(self, model, criterion_cls, metric_ftns, args, device, test_dataloader):
        super(Tester, self).__init__(model, criterion_cls, metric_ftns, args, device)
        self.test_dataloader = test_dataloader

    def test_blip(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, have_progress = [], [], []

            for batch_idx, (images, context_images, captions, cls_labels, context_cls_labels, context_ids, context_segids, context_attmasks, has_progress) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                context_images = context_images.to(self.device)
                context_ids = context_ids.to(self.device)
                context_segids = context_segids.to(self.device)
                context_attmasks = context_attmasks.to(self.device)
                has_progress = has_progress.to(self.device)

                cls_labels = cls_labels.numpy().tolist()
                context_cls_labels = context_cls_labels.numpy().tolist()

                ground_truths = captions
                reports, _, _ = self.model.generate(images, context_images, context_ids, context_segids, context_attmasks, has_progress, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)

                test_res.extend(reports)
                test_gts.extend(ground_truths)
                if batch_idx % 10 == 0:
                    print('{}/{}'.format(batch_idx, len(self.test_dataloader)))
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            test_ce = self.chexbert_metrics.compute(test_gts, test_res, have_progress)
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_ce.items()}) 
        return log, test_res, test_gts

