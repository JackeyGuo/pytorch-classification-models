""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim
from src.optim import Nadam, RMSpropTF, AdamW, RAdam, NovoGrad, NvNovoGrad, Lookahead, AdamP, SGDP
from collections.abc import Iterable
import torch.nn as nn
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def stage2_train_layer4(model, layer_names=['layer4', 'fc']):
    for name, param in model.named_parameters():
        if name.split('.')[1] in layer_names:
            continue
        param.requires_grad = False


def stage1_train_attn1(model, layer_names=['attn', 'fc']):
    for name, param in model.named_parameters():
        if name.split('.')[1] in layer_names:
            continue
        param.requires_grad = False


def stage1_train_attn(model, layer_names=['se', 'fc']):
    """
    微调layer names中的层，冻结除了layer name以外的层，BN层用eval模式
    :param model:模型
    :param layer_names:微调的层
    """
    for name, m in model.named_modules():
        # 跳过前两个不正确的module：空和module
        if name == "" or name == "module":
            continue
        # for layer_name in layer_names:
        #     if layer_name not in name.split('.'):
        #         # 冻结卷积层weight
        #         if isinstance(m, nn.Conv2d):
        #             m.weight.requires_grad = False
        #         # 如果是bn层，切换eval模式，并冻结weight和bias
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.eval()
        #             m.weight.requires_grad = False
        #             m.bias.requires_grad = False
        if layer_names[0] in name.split('.'):
            continue
        else:
            # 冻结卷积层weight
            if isinstance(m, nn.Conv2d):
                m.weight.requires_grad = False
                if m.bias is not None:
                    m.bias.requires_grad = False
            # 如果是bn层，切换eval模式，并冻结weight和bias
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def create_optimizer(args, model, filter_bias_and_bn=True, freeze_stage=""):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if 'adamw' in opt_lower or 'radam' in opt_lower:
        # Compensate for the way current AdamW and RAdam optimizers apply LR to the weight-decay
        # I don't believe they follow the paper or original Torch7 impl which schedules weight
        # decay based on the ratio of current_lr/initial_lr
        weight_decay /= args.lr
    if weight_decay and filter_bias_and_bn:
        if freeze_stage == "stage1":
            stage1_train_attn(model, layer_names=['fc'])
            print('stage1, Freeze layer successfully')
        if freeze_stage == "stage2":
            stage1_train_attn(model, layer_names=['layer3', 'layer4', 'se', 'fc'])
            stage2_train_layer4(model)
            print('stage2, Freeze layer successfully')
        # 对未冻结的层进行权重衰减
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        optimizer = optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'momentum':
        optimizer = optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=False)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'adamw':
        optimizer = AdamW(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'nadam':
        optimizer = Nadam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'radam':
        optimizer = RAdam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps,
            delta=0.1, wd_ratio=0.01, nesterov=True)
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, 
            eps=args.opt_eps, nesterov=True)        
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=weight_decay)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(
            parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=weight_decay)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusedsgd':
        optimizer = FusedSGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'fusedmomentum':
        optimizer = FusedSGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=False)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(
            parameters, lr=args.lr, adam_w_mode=False, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(
            parameters, lr=args.lr, adam_w_mode=True, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusednovograd':
        optimizer = FusedNovoGrad(
            parameters, lr=args.lr, betas=(0.95, 0.98), weight_decay=weight_decay, eps=args.opt_eps)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
