import copy
import os
import json
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import h5py
import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
import sketchgraphs.onshape.call as onshape_call


def bottom_left_pt(pts:np.array):
    s = np.sum(pts[:, 1:3], axis=1)
    return np.argmin(s), pts[np.argmin(s), 1:3]

def to_vec(seq):
    sketch = datalib.sketch_from_sequence(seq)
    entity = np.zeros((8,))
    for id, s in sketch.entities.items():
        if "isConstruction" in s.bool_ids:
            stype = -1
            _s_type = str(s.type)
            if "Line" in _s_type:
                stype = 0
                start_point = s.start_point
                end_point   = s.end_point
                _entity = np.r_[np.array(stype), start_point, end_point, np.array([-1, -1, -1])]
            elif "Arc" in _s_type:
                stype = 1
                start_point = s.start_point
                end_point   = s.end_point
                mid_point   = s.mid_point
                _entity = np.r_[np.array(stype), start_point, end_point, mid_point, np.array([-1])]
            elif "Circle" in _s_type:
                stype = 2
                center_point = s.center_point
                radius = s.radius
                _entity = np.r_[np.array(stype), center_point, np.array([-1, -1, -1, -1]), radius]
            else:
                continue
            entity = np.vstack((_entity, entity))
    loops = []
    loops.append(SOL)
    while entity.shape[0] > 0:
        start_index, start_point = bottom_left_pt(entity)
        end_point   = entity[start_index, 3:5]
        loops.append(entity[start_index, :])
        entity = np.delete(entity, start_index, 0)
        if entity[start_index, 0] == 2:
            loops.append(SOL)
            continue
        temp_loop = []
        while np.sum(end_point - start_point) > 1e-5:
            for inx, en in enumerate(entity):
                if en[0] == 2:
                    continue
                if np.sum(en[1:3] - end_point)<=1e-5 and np.sum(en[3:5] - start_point) > 1e-5:
                    temp_loop.append(en)
                    entity = np.delete(entity, inx, 0)
                    end_point = en[3:5]
                    break
                elif np.sum(en[3:5] - end_point)<=1e-5 and np.sum(en[1:3] - start_point) > 1e-5:
                    end_point = en[1:3]
                    temp = copy.deepcopy(en[1:3])
                    en[1:3] = en[3:5]
                    en[3:5] = temp
                    temp_loop.append(en)
                    entity = np.delete(entity, inx, 0)
                    break
                elif np.sum(en[1:3] - end_point) <=1e-5 and np.sum(en[3:5] - start_point)<=1e-5:
                    temp_loop.append(en)
                    entity = np.delete(entity, inx, 0)
                    end_point = en[3:5] 
                    temp_loop.append(SOL)
                    break
                elif np.sum(en[3:5] - end_point) <=1e-5 and np.sum(en[1:3] - start_point)<=1e-5:
                    end_point = en[1:3]
                    temp = copy.deepcopy(en[1:3])
                    en[1:3] = en[3:5]
                    en[3:5] = temp
                    temp_loop.append(en)
                    entity = np.delete(entity, inx, 0)
                    temp_loop.append(SOL)
                    break 
        if temp_loop == []:
            loops.pop()
            continue
        else:
            loops = loops + temp_loop
    loops = np.vstack(loops)
    end_inx = np.where(loops[:, 0] == 4)[0][-1]
    loops[end_inx] = np.array([3, -1, -1, -1, -1, -1, -1, -1])
    loop_vec = loops[:, [0, 3,4,5,6,7]]
    return loop_vec


def process(seq):
    # sketch = datalib.sketch_from_sequence(seq)
    # datalib.render_sketch(sketch)
    loop_vec = to_vec(seq)

    test_h5_path = "/comp_robot/wangcheng/vitruvion/tmp/test.h5"
    with h5py.File(test_h5_path, "w") as fp:
        fp.create_dataset('vec',  data=loop_vec, dtype=np.float32) 



if __name__ == "__main__":

    SOL = np.array([4, -1, -1, -1, -1, -1, -1, -1])
    seq_data = flat_array.load_dictionary_flat('sequence_data/data_40w/data_0.npy')
    # seq_data['sequences']

    seq = seq_data['sequences'][1327]
    process(seq)
    # print(*seq[:20], sep='\n')

    


