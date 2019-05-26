import os
import sys
import argparse
import shutil
import time
import random
import gc
import json
from distutils.version import LooseVersion
import scipy.misc
import logging

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
# import torch._utils
# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from utils.transforms import ResizeImage, ResizeAnnotation

from dataset.referit_loader import *
from model.textcam_yolo import *
from utils.parsing_metrics import *
from utils.utils import *


def yolo_loss(input, target, gi, gj, best_n_list, w_coord=5., w_neg=1./5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    batch = input[0].size(0)

    pred_bbox = Variable(torch.zeros(batch,4).cuda())
    gt_bbox = Variable(torch.zeros(batch,4).cuda())
    for ii in range(batch):
        # pred_bbox[ii, 0:2] = F.sigmoid(input[best_n_list[ii]//3][ii,best_n_list[ii]%3,0:2,gi[ii],gj[ii]])
        # pred_bbox[ii, 2:4] = input[best_n_list[ii]//3][ii,best_n_list[ii]%3,2:4,gi[ii],gj[ii]]
        # gt_bbox[ii, :] = target[best_n_list[ii]//3][ii,best_n_list[ii]%3,:4,gi[ii],gj[ii]]
        pred_bbox[ii, 0:2] = F.sigmoid(input[best_n_list[ii]//3][ii,best_n_list[ii]%3,0:2,gj[ii],gi[ii]])
        pred_bbox[ii, 2:4] = input[best_n_list[ii]//3][ii,best_n_list[ii]%3,2:4,gj[ii],gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii]//3][ii,best_n_list[ii]%3,:4,gj[ii],gi[ii]]
    loss_x = mseloss(pred_bbox[:,0], gt_bbox[:,0])
    loss_y = mseloss(pred_bbox[:,1], gt_bbox[:,1])
    loss_w = mseloss(pred_bbox[:,2], gt_bbox[:,2])
    loss_h = mseloss(pred_bbox[:,3], gt_bbox[:,3])

    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(len(input)):
        pred_conf_list.append(input[scale_ii][:,:,4,:,:].contiguous().view(batch,-1))
        gt_conf_list.append(target[scale_ii][:,:,4,:,:].contiguous().view(batch,-1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    # gt_conf = gt_conf.contiguous().view(batch,-1)
    # loss_conf = celoss(pred_conf.contiguous().view(batch,-1), gt_conf.max(1)[1])
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])

    # bce_loss = nn.BCELoss(size_average=True)
    # pred_conf = torch.clamp(F.sigmoid(pred_conf),min=1e-8,max=1-1e-8)
    # # loss_conf = -torch.mean(gt_conf * torch.log(pred_conf) \
    # #     + w_neg * (1-gt_conf) * torch.log(1-pred_conf))
    # loss_conf = bce_loss(pred_conf[gt_conf==1], gt_conf[gt_conf==1]) + \
    #     bce_loss(pred_conf[gt_conf==0], gt_conf[gt_conf==0])
    return (loss_x+loss_y+loss_w+loss_h)*w_coord + loss_conf

def save_segmentation_map(bbox, target_bbox, input, masks, mode, batch_start_index, \
    merge_pred=None, pred_conf_visu=None, save_path='./visulizations/'):
    # n, c, h, w = pred.size()
    # n, h, w = input.shape[0], input.shape[1], input.shape[2]
    n = input.shape[0]
    save_path=save_path+mode

    input=input.data.cpu().numpy()
    input=input.transpose(0,2,3,1)
    for ii in range(n):
        os.system('mkdir -p %s/sample_%d'%(save_path,batch_start_index+ii))
        imgs = input[ii,:,:,:].copy()
        imgs = (imgs*np.array([0.299, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255.
        # imgs = imgs.transpose(2,0,1)
        imgs = np.array(imgs, dtype=np.float32)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        cv2.rectangle(imgs, (bbox[ii,0], bbox[ii,1]), (bbox[ii,2], bbox[ii,3]), (255,0,0), 2)
        cv2.rectangle(imgs, (target_bbox[ii,0], target_bbox[ii,1]), (target_bbox[ii,2], target_bbox[ii,3]), (0,255,0), 2)
        cv2.imwrite('%s/sample_%d/pred_yolo.png'%(save_path,batch_start_index+ii),imgs)
        # cv2.imwrite('%s/sample_%d/pred_conf_visu.png'%(save_path,batch_start_index+ii),np.transpose(pred_conf_visu[ii,:,:]))
        cv2.imwrite('%s/sample_%d/mask.png'%(save_path,batch_start_index+ii),masks.data.cpu().numpy()[ii,:,:]*255)

def lr_poly(base_lr, iter, max_iter, power):
    # print(base_lr * ((1 - float(iter) / max_iter) ** (power)))
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def lr_step(base_lr, iter, max_iter, power):
    scale = 1.
    if iter>20:
        scale = scale/10.
        if iter>30:
            scale = scale/10.
            if iter>35:
                scale = scale/10.
    return base_lr * scale

def adjust_learning_rate(optimizer, i_iter):
    print(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
    if args.power!=0.:
        lr = lr_poly(args.lr, i_iter, args.nb_epoch, args.power)
    else:
        lr = lr_step(args.lr, i_iter, args.nb_epoch, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
      optimizer.param_groups[1]['lr'] = lr / 10
        
def save_checkpoint(state, is_best, filename='default'):
    if filename=='default':
        filename = 'textyolo_%s_batch%d'%(args.dataset,args.batch_size)

    checkpoint_name = './saved_models/%s_checkpoint.pth.tar'%(filename)
    best_name = './saved_models/%s_model_best.pth.tar'%(filename)
    torch.save(state, checkpoint_name)
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)

def build_target(raw_coord, pred):
    coord_list, bbox_list = [],[]
    for scale_ii in range(len(pred)):
        coord = Variable(torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda())
        batch, grid = raw_coord.size(0), args.size//(32//(2**scale_ii))
        coord[:,0] = (raw_coord[:,0] + raw_coord[:,2])/(2*args.size)
        coord[:,1] = (raw_coord[:,1] + raw_coord[:,3])/(2*args.size)
        coord[:,2] = (raw_coord[:,2] - raw_coord[:,0])/(args.size)
        coord[:,3] = (raw_coord[:,3] - raw_coord[:,1])/(args.size)
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0),3,5,grid, grid))

    best_n_list, best_gi, best_gj = [],[],[]

    for ii in range(batch):
        anch_ious = []
        for scale_ii in range(len(pred)):
            batch, grid = raw_coord.size(0), args.size//(32//(2**scale_ii))
            gi = coord_list[scale_ii][ii,0].long()
            gj = coord_list[scale_ii][ii,1].long()
            tx = coord_list[scale_ii][ii,0] - gi.float()
            ty = coord_list[scale_ii][ii,1] - gj.float()

            gw = coord_list[scale_ii][ii,2]
            gh = coord_list[scale_ii][ii,3]

            anchor_idxs = [x + 3*scale_ii for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            # scaled_anchors = [ (x[0] / (512./grid), x[1] / (512./grid)) for x in anchors]
            # scaled_anchors = [ (x[0] / (416./grid), x[1] / (416./grid)) for x in anchors]
            scaled_anchors = [ (x[0] / (args.size/grid), x[1] / (args.size/grid)) for x in anchors]

            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        # Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n//3
        # print(anch_ious, best_n)

        batch, grid = raw_coord.size(0), args.size//(32/(2**best_scale))
        anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
        anchors = [anchors_full[i] for i in anchor_idxs]
        # scaled_anchors = [ (x[0] / (512./grid), x[1] / (512./grid)) for x in anchors]
        # scaled_anchors = [ (x[0] / (416./grid), x[1] / (416./grid)) for x in anchors]
        scaled_anchors = [ (x[0] / (args.size/grid), x[1] / (args.size/grid)) for x in anchors]

        gi = coord_list[best_scale][ii,0].long()
        gj = coord_list[best_scale][ii,1].long()
        tx = coord_list[best_scale][ii,0] - gi.float()
        ty = coord_list[best_scale][ii,1] - gj.float()
        gw = coord_list[best_scale][ii,2]
        gh = coord_list[best_scale][ii,3]
        tw = torch.log(gw / scaled_anchors[best_n%3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n%3][1] + 1e-16)

        # bbox[ii, best_n, :, gi[ii], gj[ii]] = torch.stack([tx[ii], ty[ii], tw, th, torch.ones(1).cuda().squeeze()])
        # bbox_list[best_scale][ii, best_n%3, :, gi, gj] = torch.stack([tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
        bbox_list[best_scale][ii, best_n%3, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    for ii in range(len(bbox_list)):
        bbox_list[ii] = Variable(bbox_list[ii].cuda())

    return bbox_list, best_gi, best_gj, best_n_list

def main():
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--workers', default=16, type=int, help='num workers for data loading')
    parser.add_argument('--nb_epoch', default=40, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--power', default=0.9, type=float, help='lr poly power; 0.9-22, 0.95-44')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--size_average', dest='size_average', default=False, action='store_true', help='size_average')

    parser.add_argument('--size', default=256, type=int,
                        help='image size')
    parser.add_argument('--data_root', type=str, default='../ln_data/DMS/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='unc', type=str,
                        help='dataset used to train QSegNet unc')
    # parser.add_argument('--split', default='train', type=str,
    #                     help='name of the dataset split used to train')
    # parser.add_argument('--val', default=None, type=str,
    #                     help='name of the dataset split used to validate')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='word embedding dimensions')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--optimizer', default='RMSprop', help='optimizer: sgd, adam, RMSprop')
    parser.add_argument('--print_freq', '-p', default=500, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model, e.g. Scale_Baseline_RefineNet_batch4')
    parser.add_argument('--save_plot', dest='save_plot', default=False, action='store_true', help='save visulization plots')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--ifmfb', dest='ifmfb', default=False, action='store_true', help='ifmfb')
    parser.add_argument('--coord_emb', dest='coord_emb', default=False, action='store_true', help='coord_emb')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--query_drop', dest='query_drop', default=False, action='store_true', help='query_drop')

    global args, anchors_full
    args = parser.parse_args()
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    eps=1e-10
    if args.dataset=='refeit':
        anchors = '30,36,  78,46,  48,86,  149,79,  82,148,  331,93,  156,207,  381,163,  329,285' ## referit
    # elif args.dataset=='unc':
    #     anchors = '60,157,  95,104,  155,87,  83,187,  133,146,  113,229,  212,163,  164,251,  264,278'
    elif args.dataset=='flickr':
        anchors = '29,26,  55,58,  137,71,  82,121,  124,205,  204,132,  209,263,  369,169,  352,294'
    else:
        anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = [float(x) for x in anchors.split(',')]
    anchors_full = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)][::-1]


    if args.savename=='default':
        args.savename = 'textyolo_%s_batch%d'%(args.dataset,args.batch_size)
    logging.basicConfig(level=logging.DEBUG, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    input_transform = Compose([
        ToTensor(),
        # ResizeImage(args.size),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    target_transform = Compose([
    ])

    train_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split='train',
                         query_drop = args.query_drop,
                         imsize = args.size,
                         transform=input_transform,
                         # transform=target_transform,
                         annotation_transform=target_transform,
                         max_query_len=args.time,
                         augment=True)
    val_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split='val',
                         imsize = args.size,
                         transform=input_transform,
                         # transform=target_transform,
                         annotation_transform=target_transform,
                         max_query_len=args.time)
    ## test split to be updated accordingly
    test_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         testmode=True,
                         split='val',
                         imsize = args.size,
                         transform=input_transform,
                         # transform=target_transform,
                         annotation_transform=target_transform,
                         max_query_len=args.time)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=0)

    ## Model
    model = textcam_yolo(emb_size=args.emb_size, coordmap=True,\
        bert_model=args.bert_model, dataset=args.dataset)
    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            pretrained_dict = torch.load(args.pretrain)['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            assert (len([k for k, v in pretrained_dict.items()])!=0)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("=> loaded pretrain model at {}"
                  .format(args.pretrain))
            logging.info("=> loaded pretrain model at {}"
                  .format(args.pretrain))
        else:
            print(("=> no pretrained file found at '{}'".format(args.pretrain)))
            logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint (epoch {}) Loss{}"
                  .format(checkpoint['epoch'], best_loss)))
            logging.info("=> loaded checkpoint (epoch {}) Loss{}"
                  .format(checkpoint['epoch'], best_loss))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            logging.info(("=> no checkpoint found at '{}'".format(args.resume)))
    # print model
    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))

    visu_param = model.module.visumodel.parameters()
    rest_param = [param for param in model.parameters() if param not in visu_param]
    visu_param = list(model.module.visumodel.parameters())    ## visu_param becomes [] after rest_param
    sum_visu = sum([param.nelement() for param in visu_param])
    sum_text = sum([param.nelement() for param in model.module.textmodel.parameters()])
    sum_fusion = sum([param.nelement() for param in rest_param]) - sum_text
    # print('Num of parameters:', sum([param.nelement() for param in model.module.textmodel.parameters()]))
    # print('Num of parameters:', sum([param.nelement() for param in visu_param]))
    # print('Num of parameters:', sum([param.nelement() for param in rest_param]))
    print('visu, text, fusion parameters:', sum_visu, sum_text, sum_fusion)
    if args.optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif args.optimizer=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99)
    else:
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer = torch.optim.RMSprop([{'params': rest_param},
                {'params': visu_param, 'lr': args.lr/10.}], lr=args.lr, weight_decay=0.0005)

    best_iou = -float('Inf')
    if args.test:
        _ = test_epoch(test_loader, model, args.size_average)
        exit(0)
    for epoch in range(args.nb_epoch):
        # if args.power!=0.:
        adjust_learning_rate(optimizer, epoch)
        train_epoch(train_loader, model, optimizer, epoch, args.size_average)
        iou_new = validate_epoch(val_loader, model, args.size_average)

        ## remember best prec@1 and save checkpoint
        # is_best = iou_new < best_iou
        is_best = iou_new > best_iou
        best_iou = max(iou_new, best_iou)
        # is_best = iou_new < best_iou
        # best_iou = min(iou_new, best_iou)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': iou_new,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=args.savename)
    # print('\nBest IOU: %f\n'%best_iou)
    print('\nBest Accu: %f\n'%best_iou)
    logging.info('\nBest Accu: %f\n'%best_iou)

def train_epoch(train_loader, model, optimizer, epoch, size_average):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()
    fg_pect = AverageMeter()
    conf_matrices = AverageMeter()

    model.train()
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(train_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks.squeeze())
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        pred_anchor = model(image, word_id, word_mask)
        gt_param, gi, gj, best_n_list = build_target(bbox, pred_anchor)
        for ii in range(len(pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii].view(   \
                    pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))

        loss = yolo_loss(pred_anchor, gt_param, gi, gj, best_n_list)

        pred_coord = torch.zeros(args.batch_size,4)
        for ii in range(args.batch_size):
            best_scale_ii = best_n_list[ii]//3
            grid, grid_size = args.size//(32//(2**best_scale_ii)), 32//(2**best_scale_ii)
            anchor_idxs = [x + 3*best_scale_ii for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            # scaled_anchors = [ (x[0] / (512./grid), x[1] / (512./grid)) for x in anchors]
            # scaled_anchors = [ (x[0] / (416./grid), x[1] / (416./grid)) for x in anchors]
            scaled_anchors = [ (x[0] / (args.size/grid), x[1] / (args.size/grid)) for x in anchors]

            pred_coord[ii,0] = F.sigmoid(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 0, gj[ii], gi[ii]]) + gi[ii].float()
            pred_coord[ii,1] = F.sigmoid(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 1, gj[ii], gi[ii]]) + gj[ii].float()
            pred_coord[ii,2] = torch.exp(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 2, gj[ii], gi[ii]]) * scaled_anchors[best_n_list[ii]%3][0]
            pred_coord[ii,3] = torch.exp(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 3, gj[ii], gi[ii]]) * scaled_anchors[best_n_list[ii]%3][1]
            pred_coord[ii,:] = pred_coord[ii,:] * grid_size
        pred_coord = xywh2xyxy(pred_coord)
        # print(bbox,pred_coord,loss)

        losses.update(loss.data[0], imgs.size(0))

        target_bbox = bbox
        iou = bbox_iou(pred_coord, target_bbox.data.cpu(), x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size

        pred_conf_list, gt_conf_list = [], []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,:,4,:,:].contiguous().view(args.batch_size,-1))
            gt_conf_list.append(gt_param[ii][:,:,4,:,:].contiguous().view(args.batch_size,-1))
        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        ## if less than 3*s^2, (8*i+j)+64*b_i = max_loc
        # print(gt_conf.max(1)[1], pred_conf.max(1)[1])
        # print(pred_conf[0,gt_conf.max(1)[1]][0], pred_conf[0,pred_conf.max(1)[1]][0])
        # print(pred_conf[1,gt_conf.max(1)[1]][1], pred_conf[1,pred_conf.max(1)[1]][1])
        accu_center = np.sum(np.array(pred_conf.max(1)[1] == gt_conf.max(1)[1], dtype=float))/args.batch_size
        # print(pred_conf.shape)

        miou.update(iou.data[0], imgs.size(0))
        acc.update(accu, imgs.size(0))
        acc_center.update(accu_center, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.save_plot:
            # if batch_idx%100==0 and epoch==args.nb_epoch-1:
            if True:
                save_segmentation_map(pred_coord,target_bbox,imgs,masks,'train',batch_idx*imgs.size(0),\
                    save_path='./visulizations/%s/'%args.dataset)

        if batch_idx % args.print_freq == 0:
            # print(torch.max(attmap_out),torch.min(attmap_out))
            # # print(fvisu, flang, fvisu.shape, flang.shape)
            # print(torch.sum((attmap_out>0.5).float())/(attmap_out.view(-1).size(0)), \
            #         torch.sum((masks>0.5).float())/(masks.view(-1).size(0)))
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {acc_c.val:.4f} ({acc_c.avg:.4f})\t' \
                .format( \
                    epoch, batch_idx, len(train_loader), batch_time=batch_time, \
                    data_time=data_time, loss=losses, miou=miou, acc=acc, acc_c=acc_center) 
            print(print_str)
            logging.info(print_str)

def validate_epoch(val_loader, model, size_average, mode='val'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()
    conf_matrices = AverageMeter()

    model.eval()
    end = time.time()

    # grid, grid_size = args.size/32, 32
    # scaled_anchors = [ (x[0] / (512./grid), x[1] / (512./grid)) for x in anchors]

    for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(val_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks.squeeze())
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)
        # bbox = bbox / 2 ## bbox saved with size 512, while img 256 in exp; modify to 0-1 later 

        with torch.no_grad():
            pred_anchor = model(image, word_id, word_mask)
        for ii in range(len(pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii].view(   \
                    pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))
        gt_param, target_gi, target_gj, best_n_list = build_target(bbox, pred_anchor)

        pred_conf_list, gt_conf_list = [], []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,:,4,:,:].contiguous().view(args.batch_size,-1))
            gt_conf_list.append(gt_param[ii][:,:,4,:,:].contiguous().view(args.batch_size,-1))

        # pred_conf_visu = torch.max(F.softmax(pred_conf_list[0]).view(args.batch_size,3,args.size//32,args.size//32), dim=1)[0]
        # pred_conf_visu = F.upsample(pred_conf_visu.unsqueeze(1), scale_factor=32, mode='bilinear').squeeze()
        # pred_conf_visu = pred_conf_visu.data.cpu().numpy()*255.

        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)
        # print(gt_conf.max(1)[1], pred_conf.max(1)[1])
        # print(pred_conf[0,gt_conf.max(1)[1]][0], pred_conf[0,pred_conf.max(1)[1]][0])
        # print(pred_conf[1,gt_conf.max(1)[1]][1], pred_conf[1,pred_conf.max(1)[1]][1])

        pred_bbox = torch.zeros(args.batch_size,4)

        pred_gi, pred_gj, pred_best_n = [],[],[]
        for ii in range(args.batch_size):
            if max_loc[ii] < 3*(args.size//32)**2:
                best_scale = 0
            elif max_loc[ii] < 3*(args.size//32)**2 + 3*(args.size//16)**2:
                best_scale = 1
            else:
                best_scale = 2

            grid, grid_size = args.size//(32//(2**best_scale)), 32//(2**best_scale)
            anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            # scaled_anchors = [ (x[0] / (512./grid), x[1] / (512./grid)) for x in anchors]
            # scaled_anchors = [ (x[0] / (416./grid), x[1] / (416./grid)) for x in anchors]
            scaled_anchors = [ (x[0] / (args.size/grid), x[1] / (args.size/grid)) for x in anchors]

            pred_conf = pred_conf_list[best_scale].view(args.batch_size,3,grid,grid).data.cpu().numpy()

            # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
            (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            pred_best_n.append(best_n+best_scale*3)

            pred_bbox[ii,0] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 0, gj, gi]) + gi
            pred_bbox[ii,1] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 1, gj, gi]) + gj
            pred_bbox[ii,2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            pred_bbox[ii,3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
            pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size
        pred_bbox = xywh2xyxy(pred_bbox)
        target_bbox = bbox

        # ## Best_n
        # gt_best_n = []
        # for ii in range(args.batch_size):
        #     gw = grid * (target_bbox[ii,2] - target_bbox[ii,0])/(args.size)
        #     gh = grid * (target_bbox[ii,3] - target_bbox[ii,1])/(args.size)
        #     gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
        #     anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))
        #     anch_ious = bbox_iou(gt_box, anchor_shapes)
        #     gt_best_n.append(np.argmax(anch_ious))
        # # print(pred_best_n, gt_best_n)
        # ## gi, gj
        # coord_x = (target_bbox[:,0] + target_bbox[:,2])/(2*args.size)
        # coord_y = (target_bbox[:,1] + target_bbox[:,3])/(2*args.size)
        # target_gi = (coord_x*grid).long().data.cpu().numpy()
        # target_gj = (coord_y*grid).long().data.cpu().numpy()
        # # print(target_gi, np.array(pred_gi))
        # # print(target_gj, np.array(pred_gj))
        # print(pred_bbox, target_bbox)
        iou = bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)
        accu_center = np.sum(np.array((target_gi == np.array(pred_gi)) * (target_gj == np.array(pred_gj)), dtype=float))/args.batch_size
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size

        # losses.update(iou.data[0], imgs.size(0))
        acc.update(accu, imgs.size(0))
        acc_center.update(accu_center, imgs.size(0))
        miou.update(iou.data[0], imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        if args.save_plot:
            if batch_idx%1==0:
                save_segmentation_map(pred_bbox,target_bbox,imgs,masks,'val',batch_idx*imgs.size(0),\
                    save_path='./visulizations/%s/'%args.dataset)
        
        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {acc_c.val:.4f} ({acc_c.avg:.4f})\t' \
                .format( \
                    batch_idx, len(val_loader), batch_time=batch_time, \
                    data_time=data_time, \
                    acc=acc, acc_c=acc_center, miou=miou)
            print(print_str)
            logging.info(print_str)
    print(best_n_list, pred_best_n)
    print(np.array(target_gi), np.array(pred_gi))
    print(np.array(target_gj), np.array(pred_gj),'-')
    print(acc.avg, miou.avg,acc_center.avg)
    # logging.info(best_n_list, pred_best_n)
    # logging.info(np.array(target_gi), np.array(pred_gi))
    # logging.info(np.array(target_gj), np.array(pred_gj),'-')
    logging.info("%f,%f,%f"%(acc.avg, float(miou.avg),acc_center.avg))
    return acc.avg


def test_epoch(val_loader, model, size_average, mode='test'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()
    conf_matrices = AverageMeter()

    model.eval()
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, bbox, ratio, dw, dh, im_id) in enumerate(val_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks.squeeze())
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)
        # bbox = bbox / 2 ## bbox saved with size 512, while img 256 in exp; modify to 0-1 later 

        with torch.no_grad():
            pred_anchor = model(image, word_id, word_mask)
        for ii in range(len(pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii].view(   \
                    pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))
        gt_param, target_gi, target_gj, best_n_list = build_target(bbox, pred_anchor)

        pred_conf_list, gt_conf_list = [], []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,:,4,:,:].contiguous().view(1,-1))
            gt_conf_list.append(gt_param[ii][:,:,4,:,:].contiguous().view(1,-1))

        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)

        pred_bbox = torch.zeros(1,4)

        pred_gi, pred_gj, pred_best_n = [],[],[]
        for ii in range(1):
            if max_loc[ii] < 3*(args.size//32)**2:
                best_scale = 0
            elif max_loc[ii] < 3*(args.size//32)**2 + 3*(args.size//16)**2:
                best_scale = 1
            else:
                best_scale = 2

            grid, grid_size = args.size//(32//(2**best_scale)), 32//(2**best_scale)
            anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            # scaled_anchors = [ (x[0] / (512./grid), x[1] / (512./grid)) for x in anchors]
            # scaled_anchors = [ (x[0] / (416./grid), x[1] / (416./grid)) for x in anchors]
            scaled_anchors = [ (x[0] / (args.size/grid), x[1] / (args.size/grid)) for x in anchors]

            pred_conf = pred_conf_list[best_scale].view(1,3,grid,grid).data.cpu().numpy()

            # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
            (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            pred_best_n.append(best_n+best_scale*3)

            pred_bbox[ii,0] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 0, gj, gi]) + gi
            pred_bbox[ii,1] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 1, gj, gi]) + gj
            pred_bbox[ii,2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            pred_bbox[ii,3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
            pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size
        pred_bbox = xywh2xyxy(pred_bbox)
        target_bbox = bbox.data.cpu()
        pred_bbox[:,0], pred_bbox[:,2] = (pred_bbox[:,0]-dw)/ratio, (pred_bbox[:,2]-dw)/ratio
        pred_bbox[:,1], pred_bbox[:,3] = (pred_bbox[:,1]-dh)/ratio, (pred_bbox[:,3]-dh)/ratio
        target_bbox[:,0], target_bbox[:,2] = (target_bbox[:,0]-dw)/ratio, (target_bbox[:,2]-dw)/ratio
        target_bbox[:,1], target_bbox[:,3] = (target_bbox[:,1]-dh)/ratio, (target_bbox[:,3]-dh)/ratio

        top, bottom = round(float(dh[0]) - 0.1), args.size - round(float(dh[0]) + 0.1)
        left, right = round(float(dw[0]) - 0.1), args.size - round(float(dw[0]) + 0.1)
        img_np = imgs[0,:,top:bottom,left:right].data.cpu().numpy().transpose(1,2,0)
        mask_np = masks[top:bottom,left:right].data.cpu().numpy()

        ratio = float(ratio)
        new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)  # resized, no border
        mask_np = cv2.resize(mask_np, new_shape, interpolation=cv2.INTER_NEAREST)  # resized, no border
        img_np = Variable(torch.from_numpy(img_np.transpose(2,0,1)).cuda().unsqueeze(0))
        mask_np = Variable(torch.from_numpy(mask_np).cuda().unsqueeze(0))

        pred_bbox[:,:2], pred_bbox[:,2], pred_bbox[:,3] = \
            torch.clamp(pred_bbox[:,:2], min=0), torch.clamp(pred_bbox[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox[:,3], max=img_np.shape[2])
        target_bbox[:,:2], target_bbox[:,2], target_bbox[:,3] = \
            torch.clamp(target_bbox[:,:2], min=0), torch.clamp(target_bbox[:,2], max=img_np.shape[3]), torch.clamp(target_bbox[:,3], max=img_np.shape[2])

        iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
        accu_center = np.sum(np.array((target_gi == np.array(pred_gi)) * (target_gj == np.array(pred_gj)), dtype=float))/1
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/1

        # if accu_center==0 and accu==1:
        #     print(int(target_gi[0]),int(target_gj[0]),int(best_n_list[0]))
        #     print(pred_gi[0],pred_gj[0],pred_best_n[0],'---')
        # losses.update(iou.data[0], imgs.size(0))
        acc.update(accu, imgs.size(0))
        acc_center.update(accu_center, imgs.size(0))
        miou.update(iou.data[0], imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        if args.save_plot:
            if batch_idx%1==0:
                save_segmentation_map(pred_bbox,target_bbox,img_np,mask_np,'test',batch_idx*imgs.size(0),\
                    save_path='./visulizations/%s/'%args.dataset)
        
        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {acc_c.val:.4f} ({acc_c.avg:.4f})\t' \
                .format( \
                    batch_idx, len(val_loader), batch_time=batch_time, \
                    data_time=data_time, \
                    acc=acc, acc_c=acc_center, miou=miou)
            print(print_str)
            logging.info(print_str)
    print(best_n_list, pred_best_n)
    print(np.array(target_gi), np.array(pred_gi))
    print(np.array(target_gj), np.array(pred_gj),'-')
    print(acc.avg, miou.avg,acc_center.avg)
    # logging.info(best_n_list, pred_best_n)
    # logging.info(np.array(target_gi), np.array(pred_gi))
    # logging.info(np.array(target_gj), np.array(pred_gj),'-')
    logging.info("%f,%f,%f"%(acc.avg, float(miou.avg),acc_center.avg))
    return acc.avg


if __name__ == "__main__":
    main()
