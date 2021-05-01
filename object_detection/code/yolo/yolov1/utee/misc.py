import torch
from torch.utils import model_zoo
import torch.nn as nn
from collections import OrderedDict
import re


def load_state_dict(model, model_url, model_root):
    model_state_dict = model.state_dict()
    own_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        k = re.sub(r"group\d+\.", "", k)
        own_state_dict[k] = v

    state_dict = model_zoo.load_url(model_url, model_root)
    for k, v in state_dict.items():
        if "fc" in k:
            continue
        if k not in own_state_dict:
            raise KeyError("Unexpected keys:{}".format(k))
        if isinstance(v, nn.Parameter):
            own_state_dict[k].copy_(v.data)

    missing = set(own_state_dict.keys()) - set(state_dict.keys())
    not_used = set(state_dict.keys()) - set(own_state_dict.keys())
    print("missing keys:", missing)

    # if len(not_used) > 0:
    #     raise KeyError("Keys not used:{}".format(not_used))


def load_weights(model, pth_file):
    model_state_dict = model.state_dict()
    own_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        #k = re.sub(r"group\d+\.", "", k)
        own_state_dict[k] = v

    state_dict = torch.load(pth_file)
    for k, v in state_dict.items():
        if "module" in k:
            k = k.replace("module.", "")
        if k not in own_state_dict:
            raise KeyError("Unexpected keys:{}".format(k))
        if isinstance(v, nn.Parameter):
            own_state_dict[k].copy_(v.data)

    missing = set(own_state_dict.keys()) - set(state_dict.keys())
    not_used = set(state_dict.keys()) - set(own_state_dict.keys())
    print("missing keys:", missing)
