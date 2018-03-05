import numpy as np
import json
import tensorflow as tf


json_path = './imagenet_class_index.json'


def get_index_from_id(ids):
    f = open(json_path, 'r')
    json_dict = json.load(f)
    
    indexes = []
    for key, value in json_dict.items():
        json_id = value[0]
        if json_id in ids:
            indexes.append(int(key))
    return indexes


def decode_predictions(preds, top=5):
    _preds = np.array(preds)
    if len(preds.shape) == 1:
        _preds = np.expand_dims(_preds, axis=0)
    _preds = tf.keras.applications.resnet50.decode_predictions(_preds, top)
    ids = [p[0] for i in range(len(_preds)) for p in _preds[i]]
    ids = list(zip(*[iter(ids)]*top))
    indexes =[get_index_from_id(_ids) for _ids in ids]
    
    outputs = []
    for j in range(len(_preds)):
        o = []
        for i in range(top):
            o.append(tuple(list(_preds[j][i]) + [indexes[j][i]]))
        outputs.append(o)
    return outputs
    