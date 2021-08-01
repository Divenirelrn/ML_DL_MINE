# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def test(fasterRCNN, dataloader, imdb, roidb, logger, epoch, dataset='pascal_voc', net='res101', set_cfgs=None, load_dir='models', cuda=True, large_scale=False,
         mGPUs=True, class_agnostic=False, parallel_type=0, checksession=1, checkepoch=1, checkpoint=10021, vis=False):

  if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # np.random.seed(cfg.RNG_SEED)
  # if dataset == "pascal_voc":
  #     imdb_name = "voc_2007_trainval"
  #     imdbval_name = "voc_2007_test"
  #     set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  # elif dataset == "pascal_voc_0712":
  #     imdb_name = "voc_2007_trainval+voc_2012_trainval"
  #     imdbval_name = "voc_2007_test"
  #     set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  # elif dataset == "coco":
  #     imdb_name = "coco_2014_train+coco_2014_valminusminival"
  #     imdbval_name = "coco_2014_minival"
  #     set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  # elif dataset == "imagenet":
  #     imdb_name = "imagenet_train"
  #     imdbval_name = "imagenet_val"
  #     set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  # elif dataset == "vg":
  #     imdb_name = "vg_150-50-50_minitrain"
  #     imdbval_name = "vg_150-50-50_minival"
  #     set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  cfg_file = "cfgs/{}_ls.yml".format(net) if large_scale else "cfgs/{}.yml".format(net)

  if cfg_file is not None:
    cfg_from_file(cfg_file)
  if set_cfgs is not None:
    cfg_from_list(set_cfgs)

  #print('Using config:')
  #pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False

  print('{:d} roidb entries'.format(len(roidb)))

  #input_dir = load_dir + "/" + net + "/" + dataset
  #if not os.path.exists(input_dir):
  #  raise Exception('There is no input directory for loading network from ' + input_dir)
  #load_name = os.path.join(input_dir,
  #  'faster_rcnn_{}_{}_{}.pth'.format(checksession, checkepoch, checkpoint))

  # initilize the network here.
  #if net == 'vgg16':
  #  fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=class_agnostic)
  #elif net == 'res101':
  #  fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=class_agnostic)
  #elif net == 'res50':
  #  fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=class_agnostic)
  #elif net == 'res152':
  #  fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=class_agnostic)
  #else:
  #  print("network is not defined")
  #  pdb.set_trace()

  #fasterRCNN.create_architecture()

  #print("load checkpoint %s" % (load_name))
  #checkpoint = torch.load(load_name)
  #fasterRCNN.load_state_dict(checkpoint['model'])
  #if 'pooling_mode' in checkpoint.keys():
  #  cfg.POOLING_MODE = checkpoint['pooling_mode']

  #print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if cuda:
    cfg.CUDA = True

  #if cuda:
  #  fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  iters_per_epoch = num_images
  loss_temp = 0
  for i in range(num_images):

      data = next(data_iter)
      with torch.no_grad():
          im_data.resize_(data[0].size()).copy_(data[0])
          im_info.resize_(data[1].size()).copy_(data[1])
          gt_boxes.resize_(data[2].size()).copy_(data[2])
          #gt_boxes = gt_boxes.unsqueeze(0)
          num_boxes.resize_(data[3].size()).copy_(data[3])
          #print("test, im_data:", im_data.shape)
          #print("test, im_info:", im_info)
          #print("test, gt_boxes:", gt_boxes.shape)
          #print("test, num_boxes:", num_boxes)

          det_tic = time.time()
          rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
         
          loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
          loss_temp += loss.item()

      if i % 100 == 0:
        if i > 0:
          loss_temp /= (100 + 1)

        if mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        use_tfboard = True
        if use_tfboard:
          info = {
            'test loss': loss_temp,
            'test_loss_rpn_cls': loss_rpn_cls,
            'test_loss_rpn_box': loss_rpn_box,
            'test_loss_rcnn_cls': loss_rcnn_cls,
            'test_loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_1/test_losses", info, (epoch - 1) * iters_per_epoch + i)

        loss_temp = 0
      

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                #box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
                box_deltas = box_deltas.view(1, -1, 4)

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            #if class_agnostic:
            cls_boxes = pred_boxes[inds, :]
            #else:
            #  cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          cv2.imwrite('result.png', im2show)
          pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)

  # with open(det_file, 'wb') as f:
  #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  ap = imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))

  return ap, loss_temp
