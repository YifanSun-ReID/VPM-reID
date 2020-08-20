from __future__ import absolute_import
from .duke import Duke
from .market import Market
from .market_partial_gallery import Market_partial_gallery
from .market_partial_query import Market_partial_query
from .partial_iLIDS import Partial_iLIDS
from .partial_REID import Partial_REID
from .partial_REID_group import Partial_REID_group


__factory = {
    'market': Market,
    'duke': Duke,
    'market_partial_query': Market_partial_query,
    'market_partial_gallery': Market_partial_gallery,
    'partial_ilids': Partial_iLIDS,
    'partial_reid': Partial_REID,
    'partial_reid_group': Partial_REID_group,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'market', 'duke'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
