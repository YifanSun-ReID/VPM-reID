from __future__ import absolute_import

from .oim import oim, OIM, OIMLoss
from .triplet import TripletLoss
from .contrastive import ContrastiveLoss
from .triplet_partial import PartialTripletLoss
from .arcface import ArcFaceLoss
from .cosface import CosFaceLoss

__all__ = [
    'oim',
    'OIM',
    'OIMLoss',
    'TripletLoss',
    'Contrastive',
    'PartialTripletLoss',
    'ArcFaceLoss',
    'CosFaceLoss',
]
