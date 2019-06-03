import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
import torch.utils.data

from dataloaders.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth
from model import DepthCompletionNet
from metrics import AverageMeter, Result
import criteria
import helper
from inverse_warp import Intrinsics, homography_from
from dataloaders.kitti_loader import KittiDataset, OurDataset, NuScenesDataset, VKittiDataset
# from dataloaders.vkitti_loader import vKittiDataset

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) + ' (default: l2)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-i','--input', type=str, default='gd',
                    choices=input_options, help='input: | '.join(input_options))
parser.add_argument('-l','--layers', type=int, default=34,
                    help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('--pretrained', action="store_true",
                    help='use ImageNet pre-trained weights')
parser.add_argument('--val', type=str, default="select",
                    choices= ["select","full"], help='full or select validation set')
parser.add_argument('--jitter', type=float, default=0.1,
                    help = 'color jitter for images')
parser.add_argument('--rank-metric', type=str, default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help = 'metrics for which best result is sbatch_datacted')
parser.add_argument('-m', '--train-mode', type=str, default="dense",
                    choices = ["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
                    help = 'dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')


parser.add_argument('--dataset', action='store', dest='dataset', default='kitti',
                    type=str, nargs='?', help='dataset used')


args = parser.parse_args()
args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input) or args.use_pose
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0

dataset_name = args.dataset

print(args)

# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (args.criterion == 'l2') else criteria.MaskedL1Loss()
photometric_criterion = criteria.PhotometricLoss()
smoothness_criterion = criteria.SmoothnessLoss()

if args.use_pose:
    # hard-coded KITTI camera intrinsics
    K = load_calib()
    fu, fv = float(K[0, 0]), float(K[1, 1])
    cu, cv = float(K[0, 2]), float(K[1, 2])
    kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv).cuda()


def iterate(mode, args, loader, model, optimizer, logger, epoch, dataset):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    else:
        model.eval()
        lr = 0

    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_data = {key:val.cuda() for key,val in batch_data.items() if val is not None}
        gt = batch_data['gt'] if mode != 'test_prediction' and mode != 'test_completion' else None

        indexes = batch_data['index']
        filenames = [dataset.get_file_name(int(index.item())) for index in indexes]

        data_time = time.time() - start

        start = time.time()

        if mode == 'train':
            pred = model(batch_data)
        else:
            with torch.no_grad():
                pred = model(batch_data)

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d'])
                mask = (batch_data['d'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                depth_loss = depth_criterion(pred, gt)
                mask = (gt < 1e-3).float()

            # Loss 2: the self-supervised photometric loss
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = helper.multiscale(pred)
                rgb_curr_array = helper.multiscale(batch_data['rgb'])
                rgb_near_array = helper.multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = helper.multiscale(mask)
                num_scales = len(pred_array)

                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]

                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    intrinsics_ = kitti_intrinsics.scale(height_, width_)

                    # inverse warp from a nearby frame to the current frame
                    warped_ = homography_from(rgb_near_, pred_, batch_data['r_mat'], batch_data['t_vec'], intrinsics_)
                    photometric_loss += photometric_criterion(rgb_curr_, warped_, mask_) * (2**(scale-num_scales))

            # Loss 3: the depth smoothness loss
            smooth_loss = smoothness_criterion(pred) if args.w2>0 else 0

            # backprop
            loss = depth_loss + args.w1*photometric_loss + args.w2*smooth_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gpu_time = time.time() - start

        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.data, gt.data, photometric_loss)
            [m.update(result, gpu_time, data_time, mini_batch_size) for m in meters]
            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
            filename = filenames[0]
            logger.conditional_save_pred(mode, filename, pred, epoch)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best


def get_dataset_dir(dataset_name="kitti"):
    if dataset_name == "kitti":
        return "/home/bird/data2/dataset/kitti_depth_completion"
    elif dataset_name == "nuscenes":
        return "/home/bird/data2/dataset/nuscenes/projected"
    elif dataset_name == "ours":
        return "/home/bird/data2/dataset/our_lidar/20190315/f_c_1216_352"
    elif dataset_name == "ours_20190318":
        return "/home/bird/data2/dataset/our_lidar/20190318/f_c_1216_352"
    elif dataset_name == "vkitti":
        return "/home/bird/data2/dataset/vkitti"


def get_kitti_dataloader(mode, dataset_name, setname, args):
    """
    Get kitti dataset and dataloader according mode and setname
    :param mode: use this dataset for train or eval, possible value: train or eval
    :param dataset_name: kitti, ours, vkitti, by default, it use kitti
    :param setname: train, val, selval, test
    :param args: related arguments
    :return: dataset, dataloader
    """
    dataset_dir = get_dataset_dir(dataset_name)

    if dataset_name == 'ours':
        dataset = OurDataset(base_dir=dataset_dir, mode=mode, setname="f_c_1216_352", args=args)
    elif dataset_name == 'ours_20190318':
        dataset = OurDataset(base_dir=dataset_dir, mode=mode, setname="f_c_1216_352_20190318", args=args)
    elif dataset_name == 'vkitti':
        dataset = VKittiDataset(base_dir=dataset_dir, mode=mode, setname=setname, args=args)
    elif dataset_name == 'nuscenes':
        dataset = NuScenesDataset(base_dir=dataset_dir, mode=mode, setname="f_c_1216_352",args=args)
    elif dataset_name == 'kitti':
        dataset = KittiDataset(base_dir=dataset_dir, mode=mode, setname=setname, args=args)
    else:
        dataset = KittiDepth(setname, args)

    if mode == 'train':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True, sampler=None)
    elif mode == 'eval':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                                                 pin_memory=True)  # set batch size to be 1 for validation
    else:
        raise ValueError("Unrecognized mode " + str(mode))

    return dataset, dataloader


def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            args = checkpoint['args']
            is_eval = True
            print("=> checkpoint loaded.")
        else:
            print("=> no model found at '{}'".format(args.evaluate))
            return
    elif args.resume: # optionally resume from a checkpoint
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']+1
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer...")
    model = DepthCompletionNet(args).cuda()
    model_named_params = [p for _,p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay)
    print("=> model and optimizer created.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    model = torch.nn.DataParallel(model)
    print("=> model transferred to multi-GPU.")

    # Data loading code
    print("=> creating data loaders ...")
    if not is_eval:
        train_dataset, train_loader = get_kitti_dataloader(mode='train', dataset_name=dataset_name, setname='train', args=args)
        # train_dataset = KittiDepth('train', args)
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=args.batch_size, shuffle=True,
        #     num_workers=args.workers, pin_memory=True, sampler=None)

    val_dataset, val_loader = get_kitti_dataloader(mode='eval', dataset_name=dataset_name, setname='test', args=args)

    # change dataset here:
    # val_dataset = KittiDepth('val', args)
    # val_dataset = KittiDataset(base_dir="./data/kitti/", setname="selval")
    # val_dataset = vKittiDataset(base_dir="./data/vkitti/", setname="test")
    # val_dataset = OurDataset(base_dir="/home/bird/data2/dataset/our_lidar/20190315/f_c_1216_352", setname="f_c_1216_352")
    # val_dataset = OurDataset(base_dir="/home/bird/data2/dataset/our_lidar/20190318/f_c_1216_352", setname="f_c_1216_352_20190318")
    # val_dataset = NuScenesDataset(base_dir="/home/bird/data2/dataset/nuscenes/projected", setname="f_c_1216_352")
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #     batch_size=1, shuffle=False, num_workers=2, pin_memory=True)  # set batch size to be 1 for validation

    print("=> data loaders created.")

    # create backups and results folder
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    if is_eval:
        result, is_best = iterate("eval", args, val_loader, model, None, logger, checkpoint['epoch'], val_dataset)
        print(result)
        print(is_best)
        return


    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger, epoch, train_dataset)  # train for one epoch
        result, is_best = iterate("val", args, val_loader, model, None, logger, epoch, val_dataset)  # evaluate on validation set
        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory)


if __name__ == '__main__':
    main()
