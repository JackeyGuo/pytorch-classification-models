import argparse
import os
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from src.models import create_model, apply_test_time_pool, load_checkpoint
from src.data import Dataset, DatasetTar, create_loader, resolve_data_config, RealLabelsImagenet
from src.utils import accuracy, AverageMeter, setup_default_logging, set_jit_legacy
from torch.nn import functional as F
import shutil
import json
from eval_map import eval_map
import numpy as np


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name = "20201101-222000-ig_resnext101_32x8d-224"
dataset = "v"
if dataset == "t":
    data_path = r'/home/data/classification/action/new/test'
    json_name = "t-info"
elif dataset == "tv":
    data_path = r'/home/data/classification/action/new_data/total/valid'
    json_name = "total-v-info"
else:
    data_path = r'/home/data/classification/action/datav2/testA_clean'
    json_name = "v2-info-test"
output_path = "outputv2/%s" % model_name
parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--data', default=data_path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='%s' % model_name.split('-')[-2],
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=model_name.split('-')[-1], type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=4,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default=output_path, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=True,
                    help='disable fast prefetcher')


def validate(args):
    # might as well try to validate something
    args.prefetcher = not args.no_prefetcher

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model)
    # model, test_time_pool = apply_test_time_pool(model, data_config, args)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = Dataset(args.data)

    crop_pct = data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    f1_m = AverageMeter()

    end = time.time()
    total_pred_idx = []
    total_truth_idx = []
    mistake_image = []
    mistake_image_dict = {'calling': [], 'normal': [], 'smoking': [], 'smoking_calling': []}
    # class_2_index = {0: 'normal', 1: 'phone', 2: 'smoke'}
    class_2_index = {0: 'calling', 1: 'normal', 2: 'smoking', 3: 'smoking_calling'}
    with open("./txts/%s.json" % json_name, 'r', encoding="utf-8") as f:
        shape_dict = json.load(f)
    dets_info = {}

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + data_config['input_size'])
        if torch.cuda.is_available():
            input = input.cuda()

        model(input)
        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher and torch.cuda.is_available():
                target = target.cuda()
                input = input.cuda()

            # compute output
            # t0 = time.time()
            output = model(input)
            # print("time0: %.8f s" % ((time.time() - t0)))
            # t1 = time.time()
            # out = output.detach().cpu()
            # print("time1: %.8f s" % ((time.time() - t1) / 64))
            # print("time2: %.8f s" % ((time.time() - t0) / 64))
            # t2 = time.time()
            # out = out.cuda().cpu()
            # print("time3: %.8f s" % ((time.time() - t2) / 64))
            # get prediction index and ground turth index
            prob = torch.max(F.softmax(output, -1), -1)[0]
            idx = torch.max(F.softmax(output, -1), -1)[1]

            target_idx = target.cpu().numpy()
            predict_idx = idx.cpu().numpy()

            for j in range(len(target_idx)):
                total_truth_idx.append(target_idx[j])
                total_pred_idx.append(predict_idx[j])

                class_dict = loader.dataset.class_to_idx

                target_class = list(class_dict.keys())[list(class_dict.values()).index(int(target_idx[j]))]
                pred_class = list(class_dict.keys())[list(class_dict.values()).index(int(predict_idx[j]))]

                filename = loader.dataset.filenames()[batch_idx * args.batch_size + j]
                name = filename.split('/')[-1].split('.')[0]

                dets_info[name] = [pred_class, float(prob[j]), shape_dict[name][1], shape_dict[name][2]]

                if target_idx[j] != predict_idx[j]:
                    mistake_image.append(
                        [loader.dataset.filenames()[batch_idx * args.batch_size + j], target_class, pred_class,
                         np.round(prob[j].cpu().numpy(), 4)])

                    mistake_image_dict[class_2_index[predict_idx[j]]].append(
                        loader.dataset.filenames()[batch_idx * args.batch_size + j])

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 3))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.2f} ({top1.avg:>7.2f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))

    with open("%s/%s.json" % (output_path, json_name.split('-')[0]), "w", encoding="utf-8") as f:
        json.dump(dets_info, f)

    top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'],
        mistake_image_dict=mistake_image_dict,
        pred_idx=total_pred_idx, truth_idx=total_truth_idx)

    _logger.info(' * Acc@1 {:.2f} ({:.2f}) Acc@5 {:.2f} ({:.2f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    map, each_ap = eval_map(detFolder="%s/%s.json" % (output_path, json_name.split('-')[0]),
                   gtFolder="txts/%s.json" % json_name, return_each_ap=True)
    _logger.info('Valid mAP: {}, each ap: {}'.format(round(map, 4), each_ap))

    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    root_path = r'./results'
    result_file_path = os.path.join(root_path, args.checkpoint.split('/')[-1])
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    res_cf = open('%s/results-all.csv' % result_file_path, mode='w')

    args.checkpoint = glob.glob(args.checkpoint + '/*.pth')[0]
    r = validate(args)

    for i in range(len(r['pred_idx'])):
        res_cf.write('{0},'.format(str(r['pred_idx'][i])))
    res_cf.write('\n')
    for i in range(len(r['truth_idx'])):
        res_cf.write('{0},'.format(str(r['truth_idx'][i])))
    res_cf.flush()  # 刷新缓存区

    save_mistake_image = False
    if save_mistake_image:
        # 将分类错误的图像保存起来
        dst_root = './error/%s' % args.checkpoint.split('/')[-2]
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
        else:
            shutil.rmtree(dst_root)

        for label, img_list in r['mistake_image_dict'].items():
            for img in img_list:
                dst_path = os.path.join(dst_root, label)
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                shutil.copy(img, os.path.join(dst_path, img.split('/')[-1]))

if __name__ == '__main__':
    main()
