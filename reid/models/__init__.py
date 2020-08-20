from __future__ import absolute_import

from .resnet import *
from .resnet_rpp import resnet50_rpp
from .resnet_linear import resnet50_linear
from .resnet_pseudo import resnet50_pseudo
from .resnet_pseudo_column import resnet50_pseudo_column
from .resnet_pseudo_column_inference import resnet50_pseudo_column_inference
from .resnet_pseudo_mask import resnet50_pseudo_mask
from .resnet_pseudo_mask_orig import resnet50_pseudo_mask_orig
from .resnet_dynamic_part import resnet50_dynamic_part
from .resnet_pseudo_column_cosface import resnet50_pseudo_column_cosface
from .resnet_pseudo_column_cosface_ce import resnet50_pseudo_column_cosface_ce
from .resnet_pseudo_column_concate_cosface import resnet50_pseudo_column_concate_cosface
from .resnet_pseudo_column_concate_cosface_ce import resnet50_pseudo_column_concate_cosface_ce
from .resnet_pseudo_column_concate_cosface_detach_gt import resnet50_pseudo_column_concate_cosface_detach_gt
from .resnet_pseudo_column_concate_cosface_detach_score import resnet50_pseudo_column_concate_cosface_detach_score

__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet50_rpp': resnet50_rpp,
    'resnet50_linear': resnet50_linear,
    'resnet50_pseudo': resnet50_pseudo,
    'resnet50_pseudo_column': resnet50_pseudo_column,
    'resnet50_pseudo_column_inference': resnet50_pseudo_column_inference,
    'resnet50_pseudo_mask': resnet50_pseudo_mask,
    'resnet50_pseudo_mask_orig': resnet50_pseudo_mask_orig,
    'resnet50_dynamic_part': resnet50_dynamic_part,
    'resnet50_pseudo_column_cosface': resnet50_pseudo_column_cosface,
    'resnet50_pseudo_column_cosface_ce': resnet50_pseudo_column_cosface_ce,
    'resnet50_pseudo_column_concate_cosface': resnet50_pseudo_column_concate_cosface,
    'resnet50_pseudo_column_concate_cosface_ce': resnet50_pseudo_column_concate_cosface_ce,
    'resnet50_pseudo_column_concate_cosface_detach_gt': resnet50_pseudo_column_concate_cosface_detach_gt,
    'resnet50_pseudo_column_concate_cosface_detach_score': resnet50_pseudo_column_concate_cosface_detach_score,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
