from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.trainers_tri_pseudo_column import Trainer
#========================change the evaluator mode here================#
# from reid.evaluators_adaptive_part import Evaluator
#======================================================================#
from reid.evaluators_partial_ilids import Evaluator

from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

def get_data(name, data_dir, height, width, ratio, batch_size, workers, num_instances=8):
    root = osp.join(data_dir, name)
    root = data_dir
    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    listnormalizer = T.ListNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    
    num_classes = 752  #   plus 1 more label for the zero-padded feature

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    query_transformer = T.Compose([
        T.ContVerticalCropDiscret(height,width, ratio),
        T.ToTensor(),
        normalizer,
    ])

    query_loader = DataLoader(
        Preprocessor(dataset.query, root=osp.join(dataset.images_dir,dataset.query_path),
                     transform=query_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=osp.join(dataset.images_dir,dataset.gallery_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)


    return dataset, num_classes, query_loader, gallery_loader


def  main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join(args.logs_dir, 'log-parital-ilids-test.txt'))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset, num_classes, query_loader, gallery_loader = \
        get_data(args.dataset,  args.data_dir, args.height,
                 args.width, args.ratio, args.batch_size, args.workers
                 )


    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes,
                          cut_at_pooling=False, FCN=True, num_parts=args.num_parts)
    
    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model_dict = model.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)
#        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    model = nn.DataParallel(model).cuda()


    # Evaluator
    evaluator = Evaluator(model)
    print("Test:")
    with torch.no_grad():
        evaluator.evaluate(query_loader, gallery_loader,  dataset.query, dataset.gallery)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=128)
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--num_parts',type=int, default=6)
    parser.add_argument('--gpu', default='0', type=str)
    main(parser.parse_args())
