import os
import torch
import datetime
from tqdm import tqdm

from darknet import Darknet19

from torch.utils.data import DataLoader

from datasets.pascal_voc import VOCDataset
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from random import randint

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

torch.backends.cudnn.enabled = False

classes = 20
imdb = VOCDataset(cfg.imdb_train, cfg.DATA_DIR, cfg.batch_size, 
                  yolo_utils.preprocess_test, shuffle=False, 
                  dst_size=cfg.multi_scale_inp_size, n_classes=classes)
loader = DataLoader(imdb, batch_size=cfg.train_batch_size, shuffle=True, num_workers=5)

net = Darknet19()
net.load_from_npz(cfg.pretrained_model, num_conv=18)
print('cp model to gpu...')
net.cuda()
print('cp ends')
net.train()
print('load net succ...')
# optimizer
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

# tensorboad
use_tensorboard = cfg.use_tensorboard and SummaryWriter is not None
# use_tensorboard = False
if use_tensorboard:
    summary_writer = SummaryWriter(os.path.join(cfg.TRAIN_DIR, 'runs', cfg.exp_name))
else:
    summary_writer = None

batch_per_epoch = imdb.batch_per_epoch
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0
t = Timer()
step_cnt = 0
size_index = 0

for epoch in range(start_epoch, cfg.max_epoch):
    for iter, iter_data in enumerate(tqdm(loader, desc="Epoch {}".format(epoch))):
        batch = imdb.parse(iter_data, size_index)
        t.tic()
        im = batch['images']
        gt_boxes = batch['gt_boxes']
        gt_classes = batch['gt_classes']
        dontcare = batch['dontcare']
        orgin_im = batch['origin_im']

        # forward
        im_data = net_utils.np_to_variable(im,
                                           is_cuda=True,
                                           volatile=False).permute(0, 3, 1, 2)
        bbox_pred, iou_pred, prob_pred = net(im_data, gt_boxes, gt_classes, dontcare, size_index)

        # backward
        loss = net.loss
        bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
        iou_loss += net.iou_loss.data.cpu().numpy()[0]
        cls_loss += net.cls_loss.data.cpu().numpy()[0]
        train_loss += loss.data.cpu().numpy()[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
        step_cnt += 1
        duration = t.toc()
        if step % cfg.disp_interval == 0:
            train_loss /= cnt
            bbox_loss /= cnt
            iou_loss /= cnt
            cls_loss /= cnt
            print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
                   'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
                   (imdb.epoch, step_cnt, batch_per_epoch, train_loss, bbox_loss,
                    iou_loss, cls_loss, duration,
                    str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))  # noqa

            if summary_writer and step % cfg.log_interval == 0:
                summary_writer.add_scalar('loss_train', train_loss, step)
                summary_writer.add_scalar('loss_bbox', bbox_loss, step)
                summary_writer.add_scalar('loss_iou', iou_loss, step)
                summary_writer.add_scalar('loss_cls', cls_loss, step)
                summary_writer.add_scalar('learning_rate', lr, step)

                # plot results
                bbox_pred = bbox_pred.data[0:1].cpu().numpy()
                iou_pred = iou_pred.data[0:1].cpu().numpy()
                prob_pred = prob_pred.data[0:1].cpu().numpy()
                image = im[0]
                bboxes, scores, cls_inds = yolo_utils.postprocess(
                    bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh=0.3, size_index=size_index)
                im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)
                summary_writer.add_image('predict', im2show, step)

            train_loss = 0
            bbox_loss, iou_loss, cls_loss = 0., 0., 0.
            cnt = 0
            t.clear()
            size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
            print("image_size {}".format(cfg.multi_scale_inp_size[size_index]))

        if step > 0 and (step % imdb.batch_per_epoch == 0):
            if imdb.epoch in cfg.lr_decay_epochs:
                lr *= cfg.lr_decay
                optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                            momentum=cfg.momentum,
                                            weight_decay=cfg.weight_decay)
        
            if step % cfg.save_interval == 0:
                save_name = os.path.join(cfg.train_output_dir,
                                         '{}_{}.h5'.format(cfg.exp_name, imdb.epoch))
                net_utils.save_net(save_name, net)
                print(('save model: {}'.format(save_name)))
                step_cnt = 0

#imdb.close()
