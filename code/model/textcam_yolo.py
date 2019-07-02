from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .darknet import *
from .resnet import _ConvBatchNormReLU, _ResBlock, _ResBlockMG, Bottleneck, conv1x1, conv3x3

import argparse
import collections
import logging
import json
import re
import time
## can be commented if only use LSTM encoder
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
               input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size), 
                                 nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        if self.variable_lengths:
            input_lengths = (input_labels!=0).sum(1)  # Variable (batch, )

            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist() # list of sorted input_lengths
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist() # list of int sort_ixs, descending
            s2r = {s: r for r, s in enumerate(sort_ixs)} # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]  # list of int recover ixs
            assert max(input_lengths_list) == input_labels.size(1)

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.data.new(recover_ixs).long()  # Variable long

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        embedded = self.embedding(input_labels)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)            # (n, seq_len, word_vec_size)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)
        # forward rnn
        output, hidden = self.rnn(embedded)
        # recover
        if self.variable_lengths:
            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # (batch, max_len, hidden)
            output = output[recover_ixs]
        sent_output = []
        for ii in range(output.shape[0]):
            sent_output.append(output[ii,int(input_lengths_list[ii]-1),:])
        return torch.stack(sent_output, dim=0)

class textcam_yolo(nn.Module):
    def __init__(self, corpus=None, emb_size=256, jemb_drop_out=0.1, bert_model='bert-base-uncased', \
     coordmap=True, leaky=False, dataset=None, light=False):
        super(textcam_yolo, self).__init__()
        self.coordmap = coordmap
        self.light = light
        self.lstm = (corpus is not None)
        self.emb_size = emb_size
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024
        ## Visual model
        self.visumodel = Darknet(config_path='./model/yolov3.cfg')
        self.visumodel.load_weights('./saved_models/yolov3.weights')
        ## Text model
        if self.lstm:
            self.textdim, self.embdim=1024, 512
            self.textmodel = RNNEncoder(vocab_size=len(corpus),
                                          word_embedding_size=self.embdim,
                                          word_vec_size=self.textdim//2,
                                          hidden_size=self.textdim//2,
                                          bidirectional=True,
                                          input_dropout_p=0.2,
                                          variable_lengths=True)
        else:
            self.textmodel = BertModel.from_pretrained(bert_model)

        ## Mapping module
        self.mapping_visu = nn.Sequential(OrderedDict([
            ('0', _ConvBatchNormReLU(1024, emb_size, 1, 1, 0, 1, leaky=leaky)),
            ('1', _ConvBatchNormReLU(512, emb_size, 1, 1, 0, 1, leaky=leaky)),
            ('2', _ConvBatchNormReLU(256, emb_size, 1, 1, 0, 1, leaky=leaky))
        ]))
        self.mapping_lang = torch.nn.Sequential(
          nn.Linear(self.textdim, emb_size),
          nn.BatchNorm1d(emb_size),
          nn.ReLU(),
          nn.Dropout(jemb_drop_out),
          nn.Linear(emb_size, emb_size),
          nn.BatchNorm1d(emb_size),
          nn.ReLU(),
        )
        embin_size = emb_size*2
        if self.coordmap:
            embin_size+=8
        if self.light:
            self.fcn_emb = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    _ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('1', torch.nn.Sequential(
                    _ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('2', torch.nn.Sequential(
                    _ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ]))
            self.fcn_out = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    nn.Conv2d(emb_size, 3*5, kernel_size=1),)),
                ('1', torch.nn.Sequential(
                    nn.Conv2d(emb_size, 3*5, kernel_size=1),)),
                ('2', torch.nn.Sequential(
                    nn.Conv2d(emb_size, 3*5, kernel_size=1),)),
            ]))
        else:
            self.fcn_emb = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    _ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    _ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    _ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('1', torch.nn.Sequential(
                    _ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    _ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    _ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('2', torch.nn.Sequential(
                    _ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    _ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    _ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ]))
            self.fcn_out = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    _ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 3*5, kernel_size=1),)),
                ('1', torch.nn.Sequential(
                    _ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 3*5, kernel_size=1),)),
                ('2', torch.nn.Sequential(
                    _ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 3*5, kernel_size=1),)),
            ]))

    def forward(self, image, word_id, word_mask):
        ## Visual Module
        ## [1024, 13, 13], [512, 26, 26], [256, 52, 52]
        batch_size = image.size(0)
        raw_fvisu = self.visumodel(image)
        fvisu = []
        for ii in range(len(raw_fvisu)):
            fvisu.append(self.mapping_visu._modules[str(ii)](raw_fvisu[ii]))
            fvisu[ii] = F.normalize(fvisu[ii], p=2, dim=1)

        ## Language Module
        if self.lstm:
            max_len = (word_id != 0).sum(1).max().data[0]
            word_id = word_id[:, :max_len]
            raw_flang = self.textmodel(word_id)
        else:
            all_encoder_layers, _ = self.textmodel(word_id, \
                token_type_ids=None, attention_mask=word_mask)
            ## Sentence feature at the first position [cls]
            raw_flang = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:]\
                 + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
            ## fix bert during training
            raw_flang = raw_flang.detach()
        flang = self.mapping_lang(raw_flang)
        flang = F.normalize(flang, p=2, dim=1)

        flangvisu = []
        for ii in range(len(fvisu)):
            flang_tile = flang.view(flang.size(0), flang.size(1), 1, 1).\
                repeat(1, 1, fvisu[ii].size(2), fvisu[ii].size(3))
            if self.coordmap:
                coord = generate_coord(batch_size, fvisu[ii].size(2), fvisu[ii].size(3))
                flangvisu.append(torch.cat([fvisu[ii], flang_tile, coord], dim=1))
            else:
                flangvisu.append(torch.cat([fvisu[ii], flang_tile], dim=1))
        ## fcn
        intmd_fea, outbox = [], []
        for ii in range(len(fvisu)):
            intmd_fea.append(self.fcn_emb._modules[str(ii)](flangvisu[ii]))
            outbox.append(self.fcn_out._modules[str(ii)](intmd_fea[ii]))
        return outbox

if __name__ == "__main__":
    import sys
    import argparse
    sys.path.append('.')
    from dataset.referit_loader import *
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    from utils.transforms import ResizeImage, ResizeAnnotation
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--size', default=416, type=int,
                        help='image size')
    parser.add_argument('--data', type=str, default='../ln_data/DMS/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--split', default='train', type=str,
                        help='name of the dataset split used to train')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--emb_size', default=256, type=int,
                        help='word embedding dimensions')
    # parser.add_argument('--lang_layers', default=3, type=int,
    #                     help='number of SRU/LSTM stacked layers')

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
    ])

    refer = ReferDataset(data_root=args.data,
                         dataset=args.dataset,
                         split=args.split,
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)

    train_loader = DataLoader(refer, batch_size=2, shuffle=True,
                              pin_memory=True, num_workers=1)

    model = textcam_yolo_light(emb_size=args.emb_size)

    for batch_idx, (imgs, word_id, word_mask, bbox) in enumerate(train_loader):
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        pred_anchor_list = model(image, word_id, word_mask)
        for pred_anchor in pred_anchor_list:
            print(pred_anchor)
            print(pred_anchor.shape)
