from ml_collections import config_dict
from bbexg.models import BBEXGCodec


def get_config():
    config = config_dict.ConfigDict()
    config.model_name = BBEXGCodec
    config.dataset_name = config_dict.placeholder(str)
    config.max_num_edges = -1
    config.seed = 0
    return config
