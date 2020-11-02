#!/usr/bin/env python
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch
import cv2
import shutil
import json
from src.models import create_model, apply_test_time_pool, load_checkpoint
from src.data import Dataset, create_loader, resolve_data_config
from src.utils import AverageMeter, setup_default_logging, MyEncoder
from torch.nn import functional as F
import glob


torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_name = "20201101-222000-ig_resnext101_32x8d-224"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--data', default='/home/data/classification/action/datav2/testA', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./infer',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='%s' % model_name.split('-')[-2],
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=model_name.split('-')[-1], type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=4,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint',
                    default='outputv2/%s' % model_name, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.checkpoint = glob.glob(args.checkpoint + '/*.pth')[0]

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained)
    load_checkpoint(model, args.checkpoint)

    logging.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    args.img_size = int(args.checkpoint.split('/')[-2].split('-')[-1])
    config = resolve_data_config(vars(args), model=model)
    # model, test_time_pool = apply_test_time_pool(model, config, args)

    if torch.cuda.is_available():
        model = model.cuda()

    loader = create_loader(
        Dataset(args.data),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=False,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=config['crop_pct'])

    model.eval()

    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    scores = []
    total_pred_idx = []
    total_truth_idx = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if torch.cuda.is_available():
                input = input.cuda()
            output = model(input)

            prob = torch.max(F.softmax(output, -1), -1)[0]
            idx = torch.max(F.softmax(output, -1), -1)[1]

            total_pred_idx.extend(idx.cpu().numpy())
            total_truth_idx.extend(target.cpu().numpy())

            scores.extend(prob.cpu().numpy())
            topk_ids.extend(idx.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                logging.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    # result_file_path = os.path.join('./results', model_name)
    # if not os.path.exists(result_file_path):
    #     os.makedirs(result_file_path)

    # res_cf = open('%s/results-all.csv' % result_file_path, mode='w')
    # for i in range(len(total_pred_idx)):
    #     res_cf.write('{0},'.format(str(total_pred_idx[i])))
    # res_cf.write('\n')
    # for i in range(len(total_truth_idx)):
    #     res_cf.write('{0},'.format(str(total_truth_idx[i])))

    # dst_root = './infer/%s' % args.checkpoint.split('/')[-2]
    # if not os.path.exists(dst_root):
    #     os.makedirs(dst_root)
    # else:
    #     shutil.rmtree(dst_root)

    result_list = []
    # class_2_index = {0: 'normal', 1: 'calling', 2: 'smoking'}
    class_2_index = {0: 'calling', 1: 'normal', 2: 'smoking', 3: 'smoking_calling'}

    with open(os.path.join(args.output_dir, 'result.json'), 'w', encoding="utf-8") as out_file:
        filenames = loader.dataset.filenames()
        for i in range(len(scores)):
            filename = filenames[i].split('/')[-1]
            name = class_2_index[topk_ids[i]]
            result_data = {"image_name": str(filename), "category": name, "score": scores[i]}
            result_list.append(result_data)

            # if scores[i] > 0.95:
            # dst_path = os.path.join(dst_root, name)
            # if not os.path.exists(dst_path):
            #     os.makedirs(dst_path)
            # shutil.copy(filenames[i], os.path.join(dst_path, filename))

        json.dump(result_list, out_file, cls=MyEncoder, indent=4)


if __name__ == '__main__':
    main()
