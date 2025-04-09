import json
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import my_pre_caption
import os

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]


class generation_train(Dataset):
    def __init__(self, transform, image_root, ann_root, tokenizer, max_words=100, dataset='mimic_cxr', args=None):
        
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        self.ann = self.annotation['train']
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words      
        self.dataset = dataset
        self.args = args
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)

        has_progress = ann['has_progress']

        context_image_path = ann['context_image']
        if not has_progress:
            context_image = torch.zeros_like(image)
        else:
            context_image = Image.open(os.path.join(self.image_root, context_image_path[0])).convert('RGB')
            context_image = self.transform(context_image)
        
        cls_labels = ann['labels']
        prompt = [SCORES[l] for l in cls_labels]
        prompt = ' '.join(prompt)+' '
        caption = prompt + my_pre_caption(ann['report'], self.max_words)
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()

        context_cls_labels = ann['context_label']
        context_cls_labels = torch.from_numpy(np.array(context_cls_labels)).long()

        context = my_pre_caption(ann['context_report'], self.max_words)
        context_token = self.tokenizer(context, return_tensors="pt", padding = 'max_length', truncation=True, max_length=self.max_words)
        context_ids = torch.squeeze(context_token['input_ids'], 0)
        context_segids = torch.squeeze(context_token['token_type_ids'], 0)
        context_attmask = torch.squeeze(context_token['attention_mask'], 0)

        return image, context_image, caption, cls_labels, context_cls_labels, context_ids, context_segids, context_attmask, has_progress
    
class generation_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, tokenizer, max_words=100, split='val', dataset='mimic_cxr', args=None):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        if dataset == 'mimic_cxr':
            self.ann = self.annotation[split]
        else: # IU
            self.ann = self.annotation
        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.args = args
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)

        if 'has_progress' in ann:
            has_progress = ann['has_progress']
        else:
            has_progress = False

        if 'context_image' in ann:
            context_image_path = ann['context_image']
        else:
            context_image_path = []

        if not has_progress:
            context_image = torch.zeros_like(image)
        else:
            context_image = Image.open(os.path.join(self.image_root, context_image_path[0])).convert('RGB')
            context_image = self.transform(context_image)
        
        if not has_progress:
            context_image = torch.zeros_like(image)
        else:
            context_image = Image.open(os.path.join(self.image_root, context_image_path[0])).convert('RGB')
            context_image = self.transform(context_image)

        caption = my_pre_caption(ann['report'], self.max_words)
        cls_labels = ann['labels']
        cls_labels = torch.from_numpy(np.array(cls_labels))

        if 'context_label' in ann:
            context_cls_labels = ann['context_label']
            context_cls_labels = torch.from_numpy(np.array(context_cls_labels)).long()
        else:
            context_cls_labels = [0] * 14
            context_cls_labels = torch.from_numpy(np.array(context_cls_labels)).long()

        if 'context_report' in ann:
            context = my_pre_caption(ann['context_report'], self.max_words)
        else:
            context = my_pre_caption("[BLA]", self.max_words)

        context_token = self.tokenizer(context, return_tensors="pt", padding = 'max_length', truncation=True, max_length=self.max_words)
        context_ids = torch.squeeze(context_token['input_ids'], 0)
        context_segids = torch.squeeze(context_token['token_type_ids'], 0)
        context_attmask = torch.squeeze(context_token['attention_mask'], 0)

        return image, context_image, caption, cls_labels, context_cls_labels, context_ids, context_segids, context_attmask, has_progress
