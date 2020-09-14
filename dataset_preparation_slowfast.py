import pandas as pd
from pathlib import Path
import numpy as np
import math

root_path = Path('I:\Faegheh')
dataset_name = 'QMAR-Dataset'
motion_kind = 'Sit-Stand'
disease_kind = 'Stroke'
normal_kind = 'Normal'
annotation = 'Annotation'
capture_kind = 'COLOR'
camera_list = ['device_0', 'device_1', 'device_2', 'device_3', 'device_4', 'device_5']
fold_number = 4
clip_duration = 64

abnormal_annotation_filename = disease_kind + '-' + motion_kind + '-score.xlsx'
abnormal_annotation_filepath = root_path / dataset_name / annotation / abnormal_annotation_filename
test_filename = 'test-' + motion_kind + '-score-' + disease_kind + '-all-devices-fold' + str(fold_number) + '.xlsx'
test_filepath = root_path / dataset_name / annotation / 'test' / test_filename
normal_path = root_path / dataset_name / motion_kind / normal_kind
training_save_filename = disease_kind + '-' + motion_kind + '-training-fold' + str(fold_number) + '-64.npy'
training_save_filepath = root_path / dataset_name / '64' / training_save_filename
testing_save_filename = disease_kind + '-' + motion_kind + '-testing-fold' + str(fold_number) + '-64.npy'
testing_save_filepath = root_path / dataset_name / '64' / testing_save_filename

image_list = [[] for i in range(6)]
image_list_temp = [[] for i in range(6)]
image_list_normal = [[] for i in range(6)]
image_list_normal_temp = [[] for i in range(6)]
training_list = [[] for i in range(13)]
video_id = []
nomal_list = []
frames_count = []
testing_list = []
test_labels = []


def isClip(clip):
    clip_paths = [Path(x).parent for x in clip]
    clip_paths = [x.as_posix() for x in clip_paths]
    if len(set(clip_paths)) == 1:
        return True
    else:
        return False


df = pd.read_excel(abnormal_annotation_filepath)
df.sort_values(by='score')
# print(df)
# df_count_all = pd.read_excel('C:\\Users\Administrator\Desktop\\1111.xlsx')
# df_count_all.sort_values(by='score')
# df_count_all = df.groupby(by=['score'])['frames'].sum()
# frames_count_list = df_count_all['frames'].values.tolist()
# print(df.loc[:,'score'].value_counts().sort_index(axis=0))
df_test = pd.read_excel(test_filepath)
# df_test = pd.merge(df[['subject','folder name']], df_test, on='folder name', how='right')
# df_test.to_excel('H:\Faegheh\QMAR-Dataset\Annotation\\test\\test_sit_stand_score_Parkinson_all_devices_fold1.xlsx',index=False)


# for index, row in df_test.iterrows():
#     video_id.append([row['folder name'],row['device number']])

delete_list = df_test['folder name'].drop_duplicates().values.tolist()
print(delete_list)

df_training = df[(True ^ df['folder name'].isin(delete_list))]
print(df_training)

df_count_all = df_training.groupby(by=['score'])['frames'].sum()
df_count_all = pd.DataFrame(df_count_all)
print(df_count_all.sort_values(by='frames'))
frames_count_list = df_count_all['frames'].values.tolist()
print(frames_count_list)

target_frame_num = int(df_count_all['frames'].mean())
while target_frame_num % (clip_duration*6) != 0:
    target_frame_num += 1
print(target_frame_num)

for index, row in df_training.iterrows():
    image_list = [[] for i in range(6)]
    image_list_test = [[] for i in range(6)]
    image_list_temp = [[] for i in range(6)]
    sequence_frame_num = 0
    sequence_path = root_path / dataset_name / motion_kind / disease_kind / row['subject'] / row['folder name']
    score = row['score']
    subject_frame = row['frames']
    subject_to_all_ratio = subject_frame / frames_count_list[score - 1]
    n_compansation = math.ceil(subject_to_all_ratio * (target_frame_num - frames_count_list[score - 1]) / clip_duration)
    # print(n_compansation)
    paths = sequence_path.rglob("*.png")
    images = [x.as_posix() for x in paths]
    images = sorted(images)
    # images = [c.as_posix() for c in images]
    # print(images)
    sequence_frame_num = len(images)

    for item in images:
        for i, camera in enumerate(camera_list):
            if camera in item:
                image_list_temp[i].append(item)
    view_count = 0
    for x in image_list_temp:
        if len(x) > 0:
            view_count += 1

    # print(image_list_temp)

    for i, items in enumerate(image_list_temp):
        temp = [image_list_temp[i][k:k + clip_duration] for k in range(0, len(image_list_temp[i]), clip_duration)]
        if len(items) > 0:
            if len(items) % clip_duration != 0:
                if len(items) % clip_duration > int(clip_duration/2):
                    while len(temp[-1]) < clip_duration:
                        temp[-1].append(temp[-1][-1])
                else:
                    temp.pop(len(temp) - 1)
            if n_compansation >= 0:
                for j in range(math.ceil(n_compansation / view_count)):
                    if len(items) < clip_duration:
                        temp.append(temp[-1])
                    else:
                        if 1 + j + clip_duration < len(items):
                            slided_clip = image_list_temp[i][1 + j:1 + j + clip_duration]
                        else:
                            slided_clip = image_list_temp[i][int((1 + j + clip_duration) % len(items)):int(
                                (1 + j + clip_duration) % len(items) + clip_duration)]
                    temp.append(slided_clip)
            if n_compansation < 0:
                temp = temp[:math.ceil(target_frame_num / clip_duration / view_count * subject_to_all_ratio)]

        image_list[i] = temp

    for i in range(len(image_list)):
        for j in range(len(image_list[i])):
            training_list[row['score']].append(image_list[i][j])

image_list_temp = [[] for i in range(6)]
paths = sorted(normal_path.rglob("*.png"))
# normal_list = [x.as_posix() for x in paths]
subject_normal = [x.parent.parent.parent.parent.name for x in paths]
folder_name_normal = [x.parent.parent.parent.name for x in paths]
df_normal = pd.DataFrame({'subject': subject_normal, 'folder name': folder_name_normal})
df_normal = df_normal.drop_duplicates()
normal_test_sample = df_test[df_test['score'] == 0]
normal_list = normal_test_sample['folder name'].drop_duplicates().values.tolist()
df_normal = df_normal[(True ^ df_normal['folder name'].isin(normal_list))]
df_normal = df_normal.sample(frac=1.0)


for index, row in df_normal.iterrows():
    image_list_normal_temp = [[] for i in range(6)]
    sequence_path = root_path / dataset_name / motion_kind / normal_kind / row['subject'] / row['folder name']
    paths = sequence_path.rglob("*.png")
    images = [x.as_posix() for x in paths]
    images = sorted(images)

    for item in images:
        for i, camera in enumerate(camera_list):
            if camera in item:
                image_list_normal_temp[i].append(item)

    for i, items in enumerate(image_list_normal_temp):
        temp = [image_list_normal_temp[i][k:k + clip_duration] for k in range(0, len(image_list_normal_temp[i]), clip_duration)]
        if len(items) > 0:
            if len(items)/clip_duration < 1:
                while len(temp[-1]) < clip_duration:
                    temp[-1].append(temp[-1][-1])
            if len(items) % clip_duration != 0:
                if len(items) % clip_duration > int(clip_duration/2):
                    while len(temp[-1]) < clip_duration:
                        temp[-1].append(temp[-1][-1])
                else:
                    temp.pop(len(temp) - 1)
        image_list_normal[i].extend(temp)

for i in range(len(image_list_normal)):
    training_list[0].extend(image_list_normal[i][:int(target_frame_num/64/6)])


for i, lists in enumerate(training_list):
    if i > 0 and len(lists) != 0:
        if len(lists) <= target_frame_num / clip_duration:
            repeat = int(target_frame_num / clip_duration / len(lists) + 1)
            temp = np.array(lists)
            temp = np.tile(temp, (repeat, 1))
            training_list[i] = temp[:int(target_frame_num / clip_duration)]
        if len(lists) > target_frame_num / clip_duration:
            temp = np.array(lists)
            np.random.shuffle(temp)
            training_list[i] = temp[:int(target_frame_num / clip_duration)]

training_list = np.array(training_list)

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
dataset_training = np.array(list(zip(labels, training_list.tolist())))
np.save(training_save_filepath, dataset_training)



## generate testing file
print(df_test)
for index, row in df_test.iterrows():
    if row['score'] == 0:
        sequence_path = root_path / dataset_name / motion_kind / normal_kind / row['subject'] / row['folder name'] / row['device number']
    else:
        sequence_path = root_path / dataset_name / motion_kind / disease_kind / row['subject'] / row['folder name'] / row['device number']
    paths = sequence_path.rglob("*.png")
    images = [x.as_posix() for x in paths]
    images = sorted(images)
    splits = [images[k:k + clip_duration] for k in range(0, len(images), clip_duration)]
    if len(images) % clip_duration != 0:
        if len(items) % clip_duration > int(clip_duration/2):
            while len(splits[-1]) < clip_duration:
                splits[-1].append(splits[-1][-1])
        else:
            splits.pop(len(splits) - 1)
    testing_list.append(splits)
    test_labels.append(str(row['score']))

dataset_testing = np.array(list(zip(test_labels, testing_list)))
np.save(testing_save_filepath, dataset_testing)







