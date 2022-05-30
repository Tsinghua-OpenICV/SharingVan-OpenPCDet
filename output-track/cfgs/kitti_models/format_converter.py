import pickle as pkl
import numpy as np

root_path_car = "/home/yqjiang/project/AB3DMOT/AB3DMOT/data/KITTI-v3/pointrcnn_Car_val"
root_path_pedestrian = "/home/yqjiang/project/AB3DMOT/AB3DMOT/data/KITTI-v3/pointrcnn_Pedestrian_val"
root_path_cyclist = "/home/yqjiang/project/AB3DMOT/AB3DMOT/data/KITTI-v3/pointrcnn_Cyclist_val"
root_path_origin_car = "/home/yqjiang/project/AB3DMOT/AB3DMOT/data/KITTI_origin/pointrcnn_Car_val"
root_path_origin_cyclist = "/home/yqjiang/project/AB3DMOT/AB3DMOT/data/KITTI_origin/pointrcnn_Cyclist_val"
root_path_origin_pedestrian = "/home/yqjiang/project/AB3DMOT/AB3DMOT/data/KITTI_origin/pointrcnn_Pedestrian_val"

train_split=[0, 2, 3, 4, 5, 7, 9, 11, 17, 20]
val_split=[1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19]
set_id_trainval=[0, 154, 601, 834, 978, 1292, 1589, 1859, \
                 2659, 3049, 3852, 4146, 4519, 4597, 4937, 5043, 5419, 5628, 5773, 6112, 7171, 8008]
val_seq = [(set_id_trainval[i],set_id_trainval[i+1]) for i in val_split]
val_seq_nums = {1:446, 6:269, 8:389, 10:293, 12:77, 13:339, 14:105, 15:375, 16:208, 18:338, 19:1058}
raw = pkl.load(open('/root/OpenPCDet/output-track/cfgs/kitti_models/pointrcnn_track_v5.1-backups/default/eval/eval_with_train/epoch_3/val/result.pkl','rb'))
frame_ids = [int(x['frame_id']) for x in raw]
converter = {x:{} for x in val_split}
for i in range(len(raw)):

  for j in range(len(val_seq)):
    if val_seq[j][0] <= frame_ids[i] < val_seq[j][1]:
      seq_id = val_split[j]
      frame_id = frame_ids[i] - val_seq[j][0]
      break
  converter[seq_id][frame_id] = raw[i]

missing_record = {x:[] for x in val_split}
for key, value in converter.items():
  for i in range(val_seq_nums[key]):
    if i not in value:
      missing_record[key].append(i)


name = {'Pedestrian': 1, 'Car': 2, 'Cyclist':3}
for seq in val_split:
  seq_name = '%04d'%seq
  f_pedestrian = open(root_path_pedestrian + '/' + seq_name + '.txt', 'w+')
  f_car = open(root_path_car + '/' + seq_name + '.txt', 'w+')
  f_cyclist = open(root_path_cyclist + '/' + seq_name + '.txt', 'w+')
  f_origin_pedestrian = open(root_path_origin_pedestrian + '/' + seq_name + '.txt', 'r+')
  f_origin_car = open(root_path_origin_car + '/' + seq_name + '.txt', 'r+')
  f_origin_cyclist = open(root_path_origin_cyclist + '/' + seq_name + '.txt', 'r+')

  missing_compensate_car = {}
  missing_compensate_pedestrian = {}
  missing_compensate_cyclist = {}
  print('missing_record:', missing_record[seq])
  for line in f_origin_cyclist.readlines():
    tmp_line = line.split(',')
    if int(tmp_line[0]) in missing_record[seq]:
      if int(tmp_line[0]) not in missing_compensate_cyclist:
        missing_compensate_cyclist[int(tmp_line[0])] = [line]
      else:
        missing_compensate_cyclist[int(tmp_line[0])].append(line)

  for line in f_origin_car.readlines():
    tmp_line = line.split(',')
    if int(tmp_line[0]) in missing_record[seq]:
      if int(tmp_line[0]) not in missing_compensate_car:
        missing_compensate_car[int(tmp_line[0])] = [line]
      else:
        missing_compensate_car[int(tmp_line[0])].append(line)

  for line in f_origin_pedestrian.readlines():
    tmp_line = line.split(',')
    if int(tmp_line[0]) in missing_record[seq]:
      if int(tmp_line[0]) not in missing_compensate_pedestrian:
        missing_compensate_pedestrian[int(tmp_line[0])] = [line]
      else:
        missing_compensate_pedestrian[int(tmp_line[0])].append(line)
  print(missing_compensate_cyclist.keys())
  print(missing_compensate_pedestrian.keys())
  print(missing_compensate_car.keys())
  tmp = converter[seq]
  for i in range(val_seq_nums[seq]):
    if i in missing_record[seq]:
      cur_car = missing_compensate_car[i] if i in missing_compensate_car else []
      cur_cyclist = missing_compensate_cyclist[i] if i in missing_compensate_cyclist else []
      cur_pedestrian = missing_compensate_pedestrian[i] if i in missing_compensate_pedestrian else []
      for line in cur_car:
        f_car.write(line)
      for line in cur_cyclist:
        f_cyclist.write(line)
      for line in cur_pedestrian:
        f_pedestrian.write(line)

    else:
      cur = tmp[i]
      for j in range(cur['name'].size):
        res = [i, name[cur['name'][j]], cur['bbox'][j,0], cur['bbox'][j,1], cur['bbox'][j,2],  cur['bbox'][j,3], \
               cur['score'][j], cur['dimensions'][j][1], cur['dimensions'][j][2], cur['dimensions'][j][0], \
               cur['location'][j][0], cur['location'][j][1], cur['location'][j][2], cur['rotation_y'][j], cur['alpha'][j]]
        s = ','.join([str(res[i]) if i < 2 else '%.4f'%res[i] for i in range(len(res))])
        if cur['name'][j] == 'Car':
          f_car.write(s + '\n')
        elif cur['name'][j] == 'Pedestrian':
          f_pedestrian.write(s + '\n')
        elif cur['name'][j] == 'Cyclist':
          f_cyclist.write(s + '\n')

  f_car.close()
  f_pedestrian.close()
  f_origin_cyclist.close()
  f_origin_car.close()
  f_origin_cyclist.close()





