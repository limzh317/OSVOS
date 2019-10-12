
#from .resnet import *
from .base import *
from .fcn import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
    }
    return models[name.lower()](**kwargs)
