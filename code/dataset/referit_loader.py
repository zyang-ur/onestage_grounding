# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import sys
import cv2
import json
import uuid
import tqdm
import math
import torch
import random
import h5py
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
## Only support python2, pre-process using python3, not requried after pre-processing
# from referit import REFER
# from referit.refer import mask as cocomask
sys.path.append('.')
import utils
from utils import Corpus

import argparse
import collections
import logging
import json
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from utils.transforms import letterbox, random_affine

sys.modules['utils'] = utils

cv2.setNumThreads(0)

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

def bbox_randscale(bbox, miniou=0.75):
    w,h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    scale_shrink = (1-math.sqrt(miniou))/2.
    scale_expand = (math.sqrt(1./miniou)-1)/2.
    w1,h1 = random.uniform(-scale_expand, scale_shrink)*w, random.uniform(-scale_expand, scale_shrink)*h
    w2,h2 = random.uniform(-scale_shrink, scale_expand)*w, random.uniform(-scale_shrink, scale_expand)*h
    bbox[0],bbox[2] = bbox[0]+w1,bbox[2]+w2
    bbox[1],bbox[3] = bbox[1]+h1,bbox[3]+h2
    return bbox

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("unique_id: %s" % (example.unique_id))
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class DatasetNotFoundError(Exception):
    pass


class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')}
    }

    def __init__(self, data_root, split_root='data', dataset='referit', imsize=256,
                 transform=None, annotation_transform=None,
                 augment=False, return_idx=False, testmode=False, query_drop=False,
                 split='train', max_query_len=128, bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.imsize = imsize
        self.query_len = max_query_len
        self.corpus = Corpus()
        self.transform = transform
        self.testmode = testmode
        self.query_drop = query_drop
        self.annotation_transform = annotation_transform
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.augment=augment
        self.return_idx=return_idx

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.mask_dir = osp.join(self.dataset_root, 'mask')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif  self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        else:   ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.mask_dir = osp.join(self.dataset_root, self.dataset, 'mask')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            self.process_dataset()

        dataset_path = osp.join(self.split_root, self.dataset)
        corpus_path = osp.join(dataset_path, 'corpus.pth')
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))
        self.corpus = torch.load(corpus_path)
        # for key in self.corpus.dictionary:
        #   print key

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

        ## generate coco_minus_refer
        # if self.testmode:
        #     self.filter_image=OrderedDict()
        #     ## for each image, save all phrase-box paris
        #     for idx in range(len(self.images)):
        #         img_file, mask_file, bbox, phrase, attri = self.images[idx]
        #         if img_file in self.filter_image:
        #             self.filter_image[img_file].append([img_file])
        #         else:
        #             self.filter_image[img_file] = [img_file]
        #     self.keylist = [key for key in self.filter_image]
        #     # print(self.keylist, len(self.keylist))
        #     with open('minusrefer.txt', 'a') as f:
        #         for item in self.keylist:
        #             f.write("%s\n" % item)
        #     exit(0)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def process_dataset(self):
        if self.dataset not in self.SUPPORTED_DATASETS:
            raise DatasetNotFoundError(
                'Dataset {0} is not supported by this loader'.format(
                    self.dataset))

        dataset_folder = osp.join(self.split_root, self.dataset)
        if not osp.exists(dataset_folder):
            os.makedirs(dataset_folder)

        if self.dataset == 'referit':
            data_func = self.process_referit
        elif self.dataset == 'flickr':
            data_func = self.process_flickr
        else:
            data_func = self.process_coco

        splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        for split in splits:
            print('Processing {0}: {1} set'.format(self.dataset, split))
            data_func(split, dataset_folder)

    def process_referit(self, setname, dataset_folder):
        self.box_dict = self.referit_box()
        split_dataset = []

        query_file = osp.join(
            self.split_dir, 'referit',
            'referit_query_{0}.json'.format(setname))
        vocab_file = osp.join(self.split_dir, 'vocabulary_referit.txt')

        query_dict = json.load(open(query_file))
        im_list = query_dict.keys()

        if len(self.corpus) == 0:
            print('Saving dataset corpus dictionary...')
            corpus_file = osp.join(self.split_root, self.dataset, 'corpus.pth')
            self.corpus.load_file(vocab_file)
            torch.save(self.corpus, corpus_file)

        for name in tqdm.tqdm(im_list):
            im_filename = name.split('_', 1)[0] + '.jpg'
            if im_filename in ['19579.jpg', '17975.jpg', '19575.jpg']:
                continue
            if osp.exists(osp.join(self.im_dir, im_filename)):
                mask_mat_filename = osp.join(self.mask_dir, name + '.mat')
                mask_pth_filename = osp.join(self.mask_dir, name + '.pth')
                if osp.exists(mask_mat_filename):
                    mask = sio.loadmat(mask_mat_filename)['segimg_t'] == 0
                    mask = mask.astype(np.float64)
                    mask = torch.from_numpy(mask)
                    torch.save(mask, mask_pth_filename)
                    os.remove(mask_mat_filename)
                bbox = self.box_dict[name]
                for query in query_dict[name]:
                    split_dataset.append((im_filename, name + '.pth', bbox, query))

        output_file = '{0}_{1}.pth'.format(self.dataset, setname)
        torch.save(split_dataset, osp.join(dataset_folder, output_file))

    def process_coco(self, setname, dataset_folder):
        split_dataset = []
        vocab_file = osp.join(self.split_dir, 'vocabulary_Gref.txt')

        refer = REFER(
            self.dataset_root, **(
                self.SUPPORTED_DATASETS[self.dataset]['params']))

        refs = [refer.refs[ref_id] for ref_id in refer.refs
                if refer.refs[ref_id]['split'] == setname]

        refs = sorted(refs, key=lambda x: x['file_name'])

        if len(self.corpus) == 0:
            print('Saving dataset corpus dictionary...')
            corpus_file = osp.join(self.split_root, self.dataset, 'corpus.pth')
            self.corpus.load_file(vocab_file)
            torch.save(self.corpus, corpus_file)

        if not osp.exists(self.mask_dir):
            os.makedirs(self.mask_dir)

        for ref in tqdm.tqdm(refs):
            img_filename = 'COCO_train2014_{0}.jpg'.format(
                str(ref['image_id']).zfill(12))
            if osp.exists(osp.join(self.im_dir, img_filename)):
                h, w, _ = cv2.imread(osp.join(self.im_dir, img_filename)).shape
                seg = refer.anns[ref['ann_id']]['segmentation']
                bbox = refer.anns[ref['ann_id']]['bbox']

                rle = cocomask.frPyObjects(seg, h, w)
                mask = np.max(cocomask.decode(rle), axis=2).astype(np.float32)
                mask = torch.from_numpy(mask)
                mask_file = str(ref['ann_id']) + '.pth'
                mask_filename = osp.join(self.mask_dir, mask_file)
                if not osp.exists(mask_filename):
                    torch.save(mask, mask_filename)

                for sentence in ref['sentences']:
                    split_dataset.append((
                        img_filename, mask_file, bbox, sentence['sent']))

        output_file = '{0}_{1}.pth'.format(self.dataset, setname)
        torch.save(split_dataset, osp.join(dataset_folder, output_file))

    def process_flickr(self, setname, dataset_folder):
        # self.box_dict = self.referit_box()
        split_dataset = []

        query_file = osp.join(
            self.dataset_root,
            'lstm_phrase_features_{0}.hdf5'.format(setname))
        vocab_file = osp.join(self.dataset_root, 'vocabulary_flickr.txt')

        # query_dict = json.load(open(query_file))
        query_file = h5py.File(query_file, 'r')
        key_list = list(query_file)
        # for key in key_list[:3]:
        #     print(key, query_file[key][:], np.array(query_file[key][:]))
        im_list = [key_list[ii] for ii in range(0, len(key_list), 3)]
        # print(key_list[:10],im_list[:10], len(im_list))

        if len(self.corpus) == 0:
            print('Saving dataset corpus dictionary...')
            corpus_file = osp.join(self.split_root, self.dataset, 'corpus.pth')
            if osp.exists(vocab_file):
                self.corpus.load_file(vocab_file)
            else:
                vocab_list = ['<pad>', '<go>', '<eos>', '<unk>']
                for name in tqdm.tqdm(im_list):
                    im_filename = name + '.jpg'
                    if osp.exists(osp.join(self.im_dir, im_filename)):
                        query_matrix = np.array(query_file[name][:])
                        query_matrix = query_matrix.reshape(-1, query_matrix.shape[-1])
                        for sample_ii in range(query_matrix.shape[0]):
                            query_list = [word.decode("utf-8") for word in query_matrix[sample_ii,:] \
                                if word.decode("utf-8") not in vocab_list]
                            vocab_list += query_list
                with open(vocab_file, 'w') as f:
                    for item in vocab_list:
                        f.write("%s\n" % item)
                self.corpus.load_file(vocab_file)                
            torch.save(self.corpus, corpus_file)

        for name in tqdm.tqdm(im_list):
            im_filename = name + '.jpg'
            if osp.exists(osp.join(self.im_dir, im_filename)):
                query_matrix = np.array(query_file[name][:])
                bbox_matrix = np.array(query_file[name+'_gt'][:])
                query_matrix = query_matrix.reshape(-1, query_matrix.shape[-1])
                bbox_matrix = bbox_matrix.reshape(-1, bbox_matrix.shape[-1])
                # bbox = self.box_dict[name]
                # for query in query_dict[name]:
                for sample_ii in range(bbox_matrix.shape[0]):
                    if np.sum(bbox_matrix[sample_ii,:])==0:
                        continue
                    bbox = bbox_matrix[sample_ii,:]
                    query_list = [word.decode("utf-8") for word in query_matrix[sample_ii,:]]
                    query = ''
                    for word in query_list:
                        if word =='':
                            break
                        query = query+' '+word
                    query = query[1:]
                    # print(bbox, query)
                    split_dataset.append((im_filename, bbox, query))

        output_file = '{0}_{1}.pth'.format(self.dataset, setname)
        torch.save(split_dataset, osp.join(dataset_folder, output_file))

    def referit_box(self):
        box_file = osp.join(
            self.split_dir, 'referit',
            'referit_bbox.json')
        box_dict = json.load(open(box_file))
        return box_dict

    def pull_item(self, idx):
        if self.dataset == 'flickr':
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, mask_file, bbox, phrase, attri = self.images[idx]
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            ## Severe bug if change order; 
            ## "img_file, mask_file, bbox, phrase = self.images[idx]" is a soft copy for some reason
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path)
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        if self.dataset != 'flickr':
            mask_path = osp.join(self.mask_dir, mask_file)
            mask = torch.load(mask_path)
        else:
            mask = torch.zeros(img.shape[0], img.shape[1])
        return img, mask, phrase, bbox

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        # return 2
        return len(self.images)

    def __getitem__(self, idx):
        img, mask, phrase, bbox = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()
        if self.query_drop:
            phrase = phrase.split(' ')
            max_drop_num = min(2, len(phrase))
            for ii in range(random.randint(0,max_drop_num-1)):
                drop_idx = random.randint(0,len(phrase)-1)
                phrase[drop_idx] = '[UNK]'
            phrase = ' '.join(phrase)
        if self.augment:
            augment_flip, augment_hsv, augment_affine, bbox_aug = True,True,True,False
        # print(bbox,phrase,idx)

        ## seems a bug in torch transformation resize, so separate in advance
        h,w = img.shape[0], img.shape[1]
        mask = mask.numpy()
        if self.augment:
            if augment_flip and random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
                bbox[0], bbox[2] = w-bbox[2]-1, w-bbox[0]-1
                phrase = phrase.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
            if augment_hsv:
                fraction = 0.50
                img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)
                a = (random.random() * 2 - 1) * fraction + 1
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)
                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
            img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
            if augment_affine:
                img, mask, bbox, M = random_affine(img, mask, bbox, \
                    degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
            if bbox_aug == True:
                bbox = bbox_randscale(bbox)
        else:   ## should be inference, or specified training
            img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh

        ## Norm, to tensor
        if self.transform is not None:
            img = self.transform(img)
        examples = read_examples(phrase, idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        if self.testmode:
            return img, mask, np.array(features[0].input_ids), np.array(features[0].input_mask), \
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        if self.return_idx:
            return img, mask, np.array(features[0].input_ids), np.array(features[0].input_mask), np.array(bbox, dtype=np.float32), self.images[idx][0]
        else:
            return img, mask, np.array(features[0].input_ids), np.array(features[0].input_mask), np.array(bbox, dtype=np.float32)

    def fetch_test(self, img, query, bbox):
        ## input: img-numpy, query-string, bbox-np array x1y1x2y2
        phrase = query.lower()
        ## seems a bug in torch transformation resize, so separate in advance
        h,w = img.shape[0], img.shape[1]
        img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
        bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
        bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
        ## Norm, to tensor
        if self.transform is not None:
            img = self.transform(img)
        examples = read_examples(phrase, 0)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        if self.testmode:
            return img, np.array(features[0].input_ids), np.array(features[0].input_mask), \
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), 0
        if self.return_idx:
            return img, np.array(features[0].input_ids), np.array(features[0].input_mask), np.array(bbox, dtype=np.float32), self.images[idx][0]
        else:
            return img, np.array(features[0].input_ids), np.array(features[0].input_mask), np.array(bbox, dtype=np.float32)


def worker_init_fn(worker_id):
    # np.random.seed(13)
    np.random.seed(13 + worker_id)

def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_

def kmeans(boxes, k=9, dist=np.median):
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters

if __name__ == '__main__':
    import nltk
    import argparse
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    # from utils.transforms import ResizeImage, ResizeAnnotation
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--size', default=416, type=int,
                        help='image size')
    parser.add_argument('--data', type=str, default='../ln_data/DMS/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--dataset', default='unc', type=str,
                        help='dataset used to train QSegNet')
    parser.add_argument('--split', default='train', type=str,
                        help='name of the dataset split used to train')
    parser.add_argument('--val', default=None, type=str,
                        help='name of the dataset split used to validate')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    args = parser.parse_args()

    torch.manual_seed(13)
    np.random.seed(13)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_transform = Compose([
        ToTensor(),
        # ResizeImage(args.size),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            # std=[1., 1., 1.])
    ])
    target_transform = Compose([
        # ResizeAnnotation(args.size),
    ])

    refer = ReferDataset(data_root=args.data,
                         dataset=args.dataset,
                         split=args.split,
                         imsize=args.size,
                         transform=input_transform,
                         # transform=target_transform,
                         annotation_transform=target_transform,
                         max_query_len=args.time,
                         augment=True)

    train_loader = DataLoader(refer, batch_size=1, shuffle=True,
                  pin_memory=True, num_workers=8, worker_init_fn=worker_init_fn)
    print(len(refer))
    # if args.val:
    if True:
        refer_val = ReferDataset(data_root=args.data,
                             dataset=args.dataset,
                             split='val',
                             imsize = args.size,
                             transform=input_transform,
                             # transform=target_transform,
                             annotation_transform=target_transform,
                             max_query_len=args.time,
                             testmode=True)
        val_loader = DataLoader(refer_val, batch_size=8, shuffle=False,
                                  pin_memory=False, num_workers=0)

    # for batch_idx, (imgs, masks, words) in enumerate(train_loader):
    #   print(imgs, imgs.shape)
    #   print(masks, masks.shape)
    #   print(words)
    #   imgs = imgs.numpy()[0,:,:,:].transpose(1,2,0)
    #   imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
    #   imgs = (imgs*np.array([0.299, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255.
    #   cv2.imwrite('img.png',imgs)
    #   cv2.imwrite('mask.png',masks.numpy()[0,:,:]*255)
    #   exit(0)

    bbox_list=[]
    # for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(train_loader):
    for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(val_loader):
        # imgs = imgs.numpy()[0,:,:,:].transpose(1,2,0).copy() 
        # cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        # imgs = (imgs*np.array([0.299, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255.
        # cv2.rectangle(imgs, (bbox[0,0], bbox[0,1]), (bbox[0,2], bbox[0,3]), (255,0,0), 2)
        # cv2.imwrite('img.png',imgs)
        # cv2.imwrite('mask.png',masks.numpy()[0,:,:]*255)
        # exit(0)
        # print(imgs, imgs.shape)
        # print(masks, masks.shape)
        # print(word_id, word_id.shape)
        # print(word_mask, word_mask.shape)
        # exit(0)
        # imgs = imgs.numpy()[0,:,:,:].transpose(1,2,0)
        # imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        # imgs = (imgs*np.array([0.299, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255.
        # cv2.imwrite('img.png',imgs)
        # cv2.imwrite('mask.png',masks.numpy()[0,:,:]*255)
        # exit(0)

        # imgs = Variable(imgs)
        # masks = Variable(masks.squeeze())
        # words = Variable(words)
        bboxes = (bbox[:,2:]-bbox[:,:2]).numpy().tolist()
        for bbox in bboxes:
            bbox_list.append(bbox)
        if batch_idx%10000==0 and batch_idx!=0:
            print(batch_idx)
    for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(train_loader):
        bboxes = (bbox[:,2:]-bbox[:,:2]).numpy().tolist()
        for bbox in bboxes:
            bbox_list.append(bbox)
        if batch_idx%10000==0 and batch_idx!=0:
            print(batch_idx)
    print('loaded')
    bbox_list = np.array(bbox_list)
    # kmeans = KMeans(n_clusters=9).fit(bbox_list)
    # print(kmeans.cluster_centers_)
    clusters = kmeans(bbox_list, k=9)
    print(clusters)
