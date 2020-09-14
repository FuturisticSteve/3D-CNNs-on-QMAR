import pandas as pd
from pathlib import WindowsPath, Path
import numpy as np
import json
import math

root_path = Path('I:\Faegheh')
dataset_name = 'QMAR-Dataset'
motion_kind = 'Sit-Stand'
motion_kind_jr = 'sit_stand'
disease_kind = 'Parkinson'
normal_kind = 'Normal'
annotation = 'Annotation'
capture_kind = 'COLOR'
camera_list = ['device_0', 'device_1', 'device_2', 'device_3', 'device_4', 'device_5']
fold_number = 4
clip_duration = 16

normal_sequence_path = root_path / dataset_name / motion_kind / normal_kind
print(normal_sequence_path)
abnormal_annotation_filename = disease_kind + '-' + motion_kind + '-score.xlsx'
abnormal_annotation_filepath = root_path / dataset_name / annotation / abnormal_annotation_filename
test_filename = 'test-' + motion_kind +'-score-' + disease_kind + '-all-devices-fold' + str(fold_number) + '.xlsx'
test_filepath = root_path / dataset_name / annotation / 'test' / test_filename
original_filename = 'test_' + motion_kind_jr +'_score_' + disease_kind + '_all_devices_fold' + str(fold_number) + '.csv'
original_filepath = root_path / dataset_name / annotation / 'test' / original_filename



a = []
b = []
c = []

for item in sorted(Path(normal_sequence_path).rglob("*.png")):
    name = item.parts
    #print(name)
    if name[3]==motion_kind and name[4]==normal_kind and name[8]==capture_kind:
        a.append(name[5])
        b.append(name[6])
        c.append(item)

dict = {'subject':a, 'folder name':b, 'Path':c}
df_normal = pd.DataFrame(dict)
df_normal = df_normal.groupby(['subject', 'folder name']).count().reset_index().rename(columns={'Path': 'frames'})
print(df_normal)
#df_normal.to_excel('C:\\Users\Administrator\Desktop\\normal.xlsx')

df = pd.read_excel(abnormal_annotation_filepath).sort_values(by='score')
df_test = pd.read_csv(original_filepath)
#print(df)
df_test = df_test.rename(columns={'file name': 'folder name'})
#print(df_test)
test_list = df_test['folder name'].values.tolist()
list_split = [x.split('.')[0] for x in test_list]
#print(list_split)
folder_list = ['_'.join(x.split('_')[5:11]) for x in list_split]
#print(folder_list)
df_test['folder name'] = pd.DataFrame(folder_list)
#print(df_test)
df_test_abnormal = df_test.loc[df_test['score'] != 0]
df_test_normal = df_test.loc[df_test['score'] == 0]
df_test_abnormal = pd.merge(df[['subject','folder name']], df_test_abnormal, on='folder name', how='right')
#print(df_test_abnormal)
df_test_normal = pd.merge(df_normal[['subject','folder name']], df_test_normal, on='folder name', how='right')
#print(df_test_normal)
df_test = pd.concat([df_test_abnormal, df_test_normal], ignore_index=False)
#print(df_test)
df_test.to_excel(test_filepath, index=False)