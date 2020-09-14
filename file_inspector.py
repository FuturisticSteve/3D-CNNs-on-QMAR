import os
import pandas as pd
from pathlib import Path
import torch


root_path = Path('H:\Faegheh')
dataset_name = 'QMAR-Dataset'
motion_kind = 'Sit-Stand'
disease_kind = 'Parkinson'
normal_kind = 'Normal'
annotation = 'Annotation'
capture_kind = 'COLOR'
camera_list = ['device_0', 'device_1', 'device_2', 'device_3', 'device_4', 'device_5']
fold_number = 1
clip_duration = 16

abnormal_annotation_filename = disease_kind + '-' + motion_kind + '-score.xlsx'
abnormal_annotation_filepath = root_path / dataset_name / annotation / abnormal_annotation_filename
test_filename = 'test-' + motion_kind +'-score-' + disease_kind + '-all-devices-fold' + str(fold_number) + '.xlsx'
test_filepath = root_path / dataset_name / annotation / 'test' / test_filename
normal_path = root_path / dataset_name / motion_kind / normal_kind
training_save_filename = disease_kind + '-' + motion_kind + '-training-fold' + str(fold_number) + '.npy'
training_save_filepath = root_path / dataset_name / training_save_filename
testing_save_filename = disease_kind + '-' + motion_kind + '-testing-fold' + str(fold_number) + '.npy'
testing_save_filepath = root_path / dataset_name / testing_save_filename


###generate frame count for each sequence
dataset_path = root_path / dataset_name
print(dataset_path)
temp_info = []
annotations = []
a = []
b = []
c = []

for item in sorted(Path(dataset_path).rglob("*.png")):
    name = item.parts
    #print(name)
    if name[3]==motion_kind and name[4]==disease_kind and name[8]==capture_kind:
        a.append(name[5])
        b.append(name[6])
        c.append(item)

dict = {'subject':a, 'folder name':b, 'Path':c}
df1 = pd.DataFrame(dict)

df1 = df1.groupby(['subject', 'folder name']).count().reset_index()
print(df1)
df1.to_excel('C:\\Users\Administrator\Desktop\\123.xlsx')


###generate frame count for each label
# df = pd.read_excel('H:\Faegheh\QMAR-Dataset\Annotation\Parkinson-sit-stand-score.xlsx')
# df = df.groupby(by=['score'])['frames'].sum()
# print(df)
# df.to_excel('C:\\Users\Administrator\Desktop\\1111.xlsx')

###inspect state_dict
pth = r'I:\Project\pretrained\slowfast_weight.pth'
sta_dic = torch.load(pth)
print('.pth type:', type(sta_dic))
print('.pth len:', len(sta_dic))
print('--------------------------')
for key,value in sta_dic["state_dict"].items():
    print(key,value.size(),sep="   ")


###inspect npy files
data = np.load('H:\Faegheh\QMAR-Dataset\Stroke-Walk-testing-fold2.npy', allow_pickle=True)
dataset = []
label = []

for item in data:
    if len(item) > 0:
        for clip in item[1]:
            dataset.append(clip)
            video_id = Path(clip[0]).parent.parent.parent.name
            camera_id = Path(clip[0]).parent.parent.name
            label.append([video_id, camera_id, int(item[0])])

print()
