import copy
import os
import numpy as np
import h5py
import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from multiprocessing import Pool
import gc

def data_generator(sequences):
    for i in sequences:
        yield i


def bottom_left_pt(pts:np.array):
    s = np.sum(pts[:, 1:3], axis=1)
    return np.argmin(s), pts[np.argmin(s), 1:3]

def to_vec(sketch):
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
                _entity = np.r_[np.array(stype), center_point, center_point, np.array([-1, -1]), radius]
            else:
                continue
            entity = np.vstack((_entity, entity))
        else: continue
        
    loops = []
    loops.append(SOL)

    while entity.shape[0] > 0:
        start_index, start_point = bottom_left_pt(entity)
        end_point   = copy.deepcopy(entity[start_index, 3:5])
        loops.append(entity[start_index, :])
        entity_type = entity[start_index, 0]
        entity = np.delete(entity, start_index, 0)
        if  entity_type == 2:
            loops.append(SOL)
            continue
        temp_loop = []
        reps=0
        while np.sum(np.abs(end_point - start_point)) > 1e-5:
            if reps > entity.shape[0]:
                break
            for inx, en in enumerate(entity):
                if en[0] == 2:
                    continue
                if np.sum(np.abs(en[1:3] - end_point))<=1e-5 and np.sum(np.abs(en[3:5] - start_point)) > 1e-5:
                    temp_loop.append(en)
                    entity = np.delete(entity, inx, 0)
                    end_point = copy.deepcopy(en[3:5])
                    break
                elif np.sum(np.abs(en[3:5] - end_point))<=1e-5 and np.sum(np.abs(en[1:3] - start_point)) > 1e-5:
                    end_point = copy.deepcopy(en[1:3])
                    temp = copy.deepcopy(en[1:3])
                    en[1:3] = en[3:5]
                    en[3:5] = temp
                    temp_loop.append(en)
                    entity = np.delete(entity, inx, 0)
                    break
                elif np.sum(np.abs(en[1:3] - end_point)) <=1e-5 and np.sum(np.abs(en[3:5] - start_point)) <= 1e-5:
                    temp_loop.append(en)
                    entity = np.delete(entity, inx, 0)
                    end_point = copy.deepcopy(en[3:5])
                    temp_loop.append(SOL)
                    break
                elif np.sum(np.abs(en[3:5] - end_point)) <=1e-5 and np.sum(np.abs(en[1:3] - start_point)) <= 1e-5:
                    end_point = copy.deepcopy(en[1:3])
                    temp = copy.deepcopy(en[1:3])
                    en[1:3] = en[3:5]
                    en[3:5] = temp
                    temp_loop.append(en)
                    entity = np.delete(entity, inx, 0)
                    temp_loop.append(SOL)
                    break 
            else:
                reps+=1
                
        if temp_loop == []:
            loops.pop()
            continue
        else:
            loops = loops + temp_loop

    # print(loops)
    loops = np.vstack(loops)
    end_inx = np.where(loops[:, 0] == 4)[0][-1]
    loops[end_inx] = np.array([3, -1, -1, -1, -1, -1, -1, -1])
    # print(loops)

    loop_vec = loops[:, [0, 3,4,5,6,7]]
    return loop_vec


def process(seq, data_idx, sub_dir):
    print("processing:", str(data_idx))
    sketch = datalib.sketch_from_sequence(seq)
    # datalib.render_sketch(sketch)
    
    try:
        loop_vec = to_vec(sketch)
    except Exception as e:
        print("convect to vector faild")
        return
        
    data_idx = str(data_idx).zfill(8)
    test_h5_path = os.path.join(H5_DIR, sub_dir, data_idx + ".h5")
    truck_dir = os.path.dirname(test_h5_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)
    # test_h5_path = "/comp_robot/wangcheng/vitruvion/tmp/test.h5"
    with h5py.File(test_h5_path, "w") as fp:
        fp.create_dataset('vec',  data=loop_vec, dtype=np.float32) 



if __name__ == "__main__":

    SOL = np.array([4, -1, -1, -1, -1, -1, -1, -1])
    H5_DIR = "/comp_robot/wangcheng/vitruvion/sketch_h5"
    if not os.path.exists(H5_DIR):
        os.makedirs(H5_DIR)
    base_dir = "/comp_robot/wangcheng/vitruvion/sequence_data/data_40w"
    dirs = os.listdir(base_dir)
    dirs.sort()
    print(dirs)
    for d in dirs[2:]:
        npy_path = os.path.join(base_dir, d)
        # npy_path = 'sequence_data/data_40w/data_3.npy'
        seq_data = flat_array.load_dictionary_flat(npy_path)
        Po = Pool(processes=64, maxtasksperchild=4)
        # data_generator = data_generator(seq_data['sequences'])
        for inx, x in enumerate(seq_data['sequences']):
            Po.apply_async(process, args=(x, inx, d.split('.')[0], ))
        Po.close()
        Po.join()
        print("finish generating data")
    # seq = seq_data['sequences'][1327]
    # process(seq)
    # print(*seq[:20], sep='\n')

    


