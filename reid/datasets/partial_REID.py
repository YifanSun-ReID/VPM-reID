from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re


class Partial_REID(object):

    def __init__(self, root):

        self.images_dir = osp.join(root)
        self.query_path = 'partial_body_images'
        self.gallery_path = 'whole_body_images'

        self.query, self.gallery = [], []
        self.num_query_ids, self.num_gallery_ids = 0, 0
        self.load()

    def preprocess(self, path, relabel=False, is_probe=True):
        img_paths = glob(osp.join(self.images_dir, path, '*.jpg'))
        all_pids = {}
        ret = []
        for img_path in img_paths:
            pid = int(osp.basename(img_path).split('_')[0])
            if is_probe:
                cam = 0
            else:
                cam = 1
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            ret.append((img_path, pid, cam))
        
        return ret, len(all_pids)
    
    def load(self):
        self.query, self.num_query_ids = self.preprocess(self.query_path, False, True)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))