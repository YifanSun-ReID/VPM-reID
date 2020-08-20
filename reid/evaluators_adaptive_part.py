from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_part_feature
from .utils.meters import AverageMeter


def extract_features(model, data_loader, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    part_score = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_part_feature(model, imgs, return_mask = False)
        for fname, output, score, pid in zip(fnames, outputs[0], outputs[1], pids):
            features[fname] = output
            part_score[fname] = score
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, part_score, labels


def pairwise_distance(query_features, gallery_features, query_parts, gallery_parts, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(1) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    px = torch.cat([query_parts[f].unsqueeze(0) for f, _, _ in query], 0)
    py = torch.cat([gallery_parts[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    num_parts = x.size(2)
    px = px.unsqueeze(1).expand(m, n, num_parts)
    py = py.unsqueeze(0).expand(m, n, num_parts)

    pjoin = px * py
    weights = pjoin / pjoin.sum(2,True).expand_as(pjoin)
    
    x = x.chunk(x.size(2),2)
    y = y.chunk(y.size(2),2)
    part_x = []
    part_y = []
    for tmp in x:
        part_x.append(tmp.squeeze(3).squeeze(2))
    for tmp in y:
        part_y.append(tmp.squeeze(3).squeeze(2))
        
    dist = []
    for i, local_x in enumerate(part_x):
        local_y = part_y[i]
        tmp = torch.pow(local_x, 2).sum(1).unsqueeze(1).expand(m, n) + \
            torch.pow(local_y, 2).sum(1).unsqueeze(1).expand(n, m).t()
        dist.append(tmp.addmm_(1, -2, local_x, local_y.t()).unsqueeze(2))
    dist = torch.cat(dist, 2)    
    
    final_dist = (dist*weights.float()).sum(2)


    return final_dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['allshots'][0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery):
        print('extracting query features\n')
        query_features, query_parts, _ = extract_features(self.model, query_loader)
        print('extracting gallery features\n')
        gallery_features, gallery_parts, _ = extract_features(self.model, gallery_loader)
        distmat = pairwise_distance(query_features, gallery_features, query_parts, gallery_parts,  query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery)
