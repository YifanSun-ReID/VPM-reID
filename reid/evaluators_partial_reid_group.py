from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap, mean_ap_partial, mean_ap_part
from .feature_extraction import extract_cnn_feature, extract_part_feature
from .utils.meters import AverageMeter
import numpy as np
import pdb



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

        outputs = extract_part_feature(model, imgs, return_mask = False)   # outputs ==> [torch.size(b, 2048, 6, 1), torch.size(b, 6)] outputs[1]代表属于某个part的概率
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

    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)  # 3368， 2048， 6， 1
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)  # 15913, 2048, 6, 1
    px = torch.cat([query_parts[f].unsqueeze(0) for f, _, _ in query], 0)   # 3368, 6
    py = torch.cat([gallery_parts[f].unsqueeze(0) for f, _, _ in gallery], 0) # 15913, 6
    m, n = x.size(0), y.size(0)
    num_parts = x.size(2)
    px = px.unsqueeze(1).expand(m, n, num_parts)
    py = py.unsqueeze(0).expand(m, n, num_parts)

    pjoin = px * py  # 对应的共同 parts 的矩阵
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




def pairwise_distance_partial(query_features, gallery_features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(1) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0) # b, 2048, 24
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0) # 15913, 2048, 24

    # x = torch.mean(x, dim=2, keepdim=False)
    # x = x / x.norm(2, 1, keepdim=True).expand_as(x)
    ratios = [1, 5/6, 4/6, 3/6, 4/5, 3/5, 3/4]
    h = x.size(2)
    x_crop = []
    for ratio in ratios:
        x_crop.append(torch.mean(x[:, :, :int(ratio * h)], dim=2, keepdim=False))
    x_crop = [f / f.norm(2, 1, keepdim=True).expand_as(f) for f in x_crop]

    # ratios = [1, 5/6, 4/6, 3/6, 4/5, 3/5, 3/4]
    y_crop = []
    for ratio in ratios:
        y_crop.append(torch.mean(y[:, :, :int(ratio * h)], dim=2, keepdim=False))
    y_crop = [f / f.norm(2, 1, keepdim=True).expand_as(f) for f in y_crop]   # [(b, 2048)] * 6
    dist_mat = []
    for i in range(len(x_crop)):
        for j in range(len(y_crop)):
            dist_mat.append(euclidean_dist(x_crop[i], y_crop[j]))
    return dist_mat 


def pairwise_distance_pcb(query_features, gallery_features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(1) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    x_list = list(x.chunk(6, 2))
    y_list = list(y.chunk(6, 2))
    x_list.append(x)
    y_list.append(y)
    x_list = [x.mean(2) for x in x_list]
    y_list = [y.mean(2) for y in y_list]
    dist = []
    for i in range(len(x_list)):
        dist.append(euclidean_dist(x_list[i], y_list[i]))
    distmat = 0
    for i in range(len(dist)):
       distmat += dist[i]
    return distmat


def pairwise_distance_part(query_features, gallery_features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(1) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)

    x_parts, y_parts = {}, {}
    for i in range(3, 7):
        x_list = x.chunk(i, 2)
        y_list = y.chunk(i, 2)
        x_parts[i] = [z.mean(2) for z in x_list]
        y_parts[i] = [z.mean(2) for z in y_list]
    
    distmat = {}
    for key, value in x_parts.items():
        distmat[key] = 0
        for i in range(len(value)):
            distmat[key] += euclidean_dist(value[i], y_parts[6][i])

    return distmat


    
def pairwise_distance_ilids(query_features, gallery_features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(1) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist
    if query is not None and gallery is not None:
        query_ids = torch.tensor([pid for _, pid, _ in query])
        gallery_ids = torch.tensor([pid for _, pid, _ in gallery])
    
    pos_indices = (gallery_ids == query_ids.unsqueeze(1)).type(torch.float)

    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    y_list = [f.mean(2) for f in y.chunk(6, 2)]

    distmat = []
    for i in range(x.size(0)):
        single = x[i].unsqueeze(0)
        pos_index = pos_indices[i]
        dist = []
        for j in range(3, 7):
            single_parts = [f.mean(2) for f in single.chunk(j, 2)]
            dist_parts = 0
            for z in range(len(single_parts)):
                dist_parts += euclidean_dist(single_parts[z], y_list[z])
            dist.append(dist_parts / j)
        pos_dist = []
        for j in range(len(dist)):
            pos_dist.append((dist[j] * pos_index).sum())
        min_index = pos_dist.index(min(pos_dist))
        distmat.append(dist[min_index])
    distmat = torch.cat(distmat, 0)

    return distmat



def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist



def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 3, 10)):
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
    # print('Mean AP: {:4.1%}'.format(mAP))

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

    # print('CMC Scores{:>12}{:>12}{:>12}'
    #       .format('allshots', 'cuhk03', 'market1501'))
    # for k in cmc_topk:
    #     print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
    #           .format(k, cmc_scores['allshots'][k - 1],
    #                   cmc_scores['cuhk03'][k - 1],
    #                   cmc_scores['market1501'][k - 1]))
    # Use the allshots cmc top-1 score for validation criterion
    # return cmc_scores['allshots'][0]
    return mAP, cmc_scores


def evaluate_all_partial(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 3, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP, distmat = mean_ap_partial(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
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


def evaluate_all_part(distmat, query=None, gallery=None,
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
    mAP, distmat = mean_ap_part(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
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

    def evaluate(self, query_loader_set, gallery_loader_set, dataset_set):
        mAP_total = []
        cmc_total = []
        for i in range(5):
            for j in range(5):
                # print('extracting query features\n')
                query_features, query_parts, _ = extract_features(self.model, query_loader_set[i])
                # print('extracting gallery features\n')
                gallery_features, gallery_parts, _ = extract_features(self.model, gallery_loader_set[j])
                query = dataset_set[i].query
                gallery = dataset_set[j].gallery
                distmat = pairwise_distance(query_features, gallery_features, query_parts, gallery_parts, query, gallery)
                mAP, cmc = evaluate_all(distmat, query=query, gallery=gallery)
                mAP_total.append(mAP)
                cmc_total.append(cmc['market1501'])
        mAP_total = np.array(mAP_total)
        mAP_total = np.mean(mAP_total)
        cmc_total = np.array(cmc_total)
        cmc_total = np.mean(cmc_total, axis=0)
        print('Mean AP: {:4.1%}'.format(mAP_total))
        cmc_topk = (1, 3, 10)
        for k in cmc_topk:
            print('CMC curve, Rank-{:<3}:{:.1%}'.format(k, cmc_total[k - 1]))
        return 