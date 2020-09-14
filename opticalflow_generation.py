"""
Created on Sun Aug 18 23:10:59 2019

@author: Abrar Istiak Akib
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import time

root_path = Path(r"I:\Faegheh\QMAR-Dataset\Sit-Stand\Parkinson\Erick\2019_3_13_12_5_49\device_5")
dataset_name = 'QMAR-Dataset'
motion_kind = 'Sit-Stand'
disease_kind = 'Parkinson'
normal_kind = 'Normal'
annotation = 'Annotation'
capture_kind = 'COLOR'
camera_list = ['device_0', 'device_1', 'device_2', 'device_3', 'device_4', 'device_5']


sequence_folders_all = [x.parent.as_posix() for x in root_path.rglob("*.png")]
sequence_folders_all = list(set(sequence_folders_all))
excpet_list = [x.parent.as_posix() for x in Path('I:\Optical Flow-tvl1').rglob("*.npy")]
excpet_list = list(set(excpet_list))
sequence_folders = []
for item in sequence_folders_all:
    if item not in excpet_list:
        sequence_folders.append(item)
print(len(sequence_folders))

paths = []
for folder_path in sequence_folders:
    paths.append([x.as_posix() for x in Path(folder_path).rglob("*.png")])

save_paths = []
for folder_path in sequence_folders:
    flow_path = 'I:\Optical Flow-tvl1'
    path_split = folder_path.split('/')
    for piece in path_split[2:]:
        flow_path = flow_path + '/' + piece
    save_paths.append(flow_path)

optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

sample_frame = cv2.imread(r"I:\Faegheh\QMAR-Dataset\Sit-Stand\Normal\Abel\2019_3_12_16_35_28\device_0\COLOR\color00287.png")
hsvx = np.zeros_like(sample_frame)
hsvx[..., 1] = 10
hsvx[..., 0] = 10
hsvy = np.zeros_like(sample_frame)
hsvy[..., 1] = 10
hsvy[..., 0] = 10

count = 1
for i, image_paths in enumerate(paths):
    start_time = time.time()

    print('Processing: '+folder_path)
    Path(save_paths[i]).mkdir(parents=True,exist_ok=True)

    images = []
    for image_path in image_paths:
        images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

    prvs = images[0]
    output= []
    output1 = []

    ii = 0
    for image in images[1:]:
        ii += 1

        try:
            flow = optical_flow.calc(prvs, image, None)
        except:
            print('ii:'+str(ii))
            continue
        #flow = cv2.calcOpticalFlowPyrLK(prvs,image,output,output1)

        hsvx[..., 2] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        rgbx = cv2.cvtColor(hsvx, cv2.COLOR_HSV2BGR)
        #rgbx = cv2.resize(cv2.cvtColor(rgbx, cv2.COLOR_BGR2GRAY),(112,112))
        hsvy[..., 2] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        rgby = cv2.cvtColor(hsvy, cv2.COLOR_HSV2BGR)
        #rgby = cv2.resize(cv2.cvtColor(rgby, cv2.COLOR_BGR2GRAY),(112,112))

        # save optical flow as 2 channel in .npy file
        # flow_file = np.stack((rgbx, rgby), axis=-1)
        # np.save(os.path.join(save_paths[i], 'frame0000'+str(ii)+'.npy'), flow_file)
        subpath_v = "C:\\Users\Administrator\Desktop\\v"
        subpath_u = "C:\\Users\Administrator\Desktop\\u"

        cv2.imwrite(os.path.join(subpath_v, 'frame0000' + str(ii) + '.jpg'), rgby)
        cv2.imwrite(os.path.join(subpath_u, 'frame0000' + str(ii) + '.jpg'), rgbx)

        prvs = image

    print('Saved to: '+save_paths[i])
    print('['+str(count)+']'+'Finished in: '+str(round(time.time()-start_time,4))+'s')
    count += 1

# root_path = Path('I:\Faegheh\QMAR-Dataset\Sit-Stand\\Normal')
# dataset_name = 'QMAR-Dataset'
# motion_kind = 'Sit-Stand'
# disease_kind = 'Parkinson'
# normal_kind = 'Normal'
# annotation = 'Annotation'
# capture_kind = 'COLOR'
# camera_list = ['device_0', 'device_1', 'device_2', 'device_3', 'device_4', 'device_5']
#
#
# sequence_folders_all = [x.parent.as_posix() for x in root_path.rglob("*.png")]
# sequence_folders_all = list(set(sequence_folders_all))
# excpet_list = [x.parent.as_posix() for x in Path('I:\Optical Flow-tvl1').rglob("*.npy")]
# excpet_list = list(set(excpet_list))
# sequence_folders = []
# for item in sequence_folders_all:
#     if item not in excpet_list:
#         sequence_folders.append(item)
# print(len(sequence_folders))
#
# paths = []
# for folder_path in sequence_folders:
#     paths.append([x.as_posix() for x in Path(folder_path).rglob("*.png")])
#
# save_paths = []
# for folder_path in sequence_folders:
#     flow_path = 'I:\Optical Flow-tvl1'
#     path_split = folder_path.split('/')
#     for piece in path_split[2:]:
#         flow_path = flow_path + '/' + piece
#     save_paths.append(flow_path)
#
# optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
#
# sample_frame = cv2.imread(r"I:\Faegheh\QMAR-Dataset\Sit-Stand\Normal\Abel\2019_3_12_16_35_28\device_0\COLOR\color00287.png")
# hsvx = np.zeros_like(sample_frame)
# hsvx[..., 1] = 10
# hsvx[..., 0] = 10
# hsvy = np.zeros_like(sample_frame)
# hsvy[..., 1] = 10
# hsvy[..., 0] = 10
#
# count = 1
# for i, image_paths in enumerate(paths):
#     start_time = time.time()
#
#     print('Processing: '+folder_path)
#     Path(save_paths[i]).mkdir(parents=True,exist_ok=True)
#
#     images = []
#     for image_path in image_paths:
#         images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
#
#     prvs = images[0]
#     output= []
#     output1 = []
#
#     ii = 0
#     for image in images[1:]:
#         ii += 1
#
#         try:
#             flow = optical_flow.calc(prvs, image, None)
#         except:
#             print('ii:'+str(ii))
#             continue
#         #flow = cv2.calcOpticalFlowPyrLK(prvs,image,output,output1)
#
#         hsvx[..., 2] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
#         rgbx = cv2.cvtColor(hsvx, cv2.COLOR_HSV2BGR)
#         rgbx = cv2.resize(cv2.cvtColor(rgbx, cv2.COLOR_BGR2GRAY),(112,112))
#         hsvy[..., 2] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
#         rgby = cv2.cvtColor(hsvy, cv2.COLOR_HSV2BGR)
#         rgby = cv2.resize(cv2.cvtColor(rgby, cv2.COLOR_BGR2GRAY),(112,112))
#
#         # save optical flow as 2 channel in .npy file
#         # flow_file = np.stack((rgbx, rgby), axis=-1)
#         # np.save(os.path.join(save_paths[i], 'frame0000'+str(ii)+'.npy'), flow_file)
#
#         subpath_v = "C:\\Users\Administrator\Desktop\\v"
#         subpath_u = "C:\\Users\Administrator\Desktop\\u"
#
#         cv2.imwrite(os.path.join(subpath_v, 'frame0000' + str(ii) + '.jpg'), rgby)
#         cv2.imwrite(os.path.join(subpath_u , 'frame0000'+str(ii)+'.jpg'), rgbx)
#
#         prvs = image
#
#     print('Saved to: '+save_paths[i])
#     print('['+str(count)+']'+'Finished in: '+str(round(time.time()-start_time,4))+'s')
#     count += 1


# #to take video from an input directory
# def video_from_dir(dir):
#     temp=os.listdir(dir)
#     video=[]
#     for i in temp:
#         if(i.endswith('.MP4')):
#             video.append(i)
#     return video
#
#
# store_root_path = root_path.parent
#
# cur=os.getcwd()
# cls_list_dir=os.path.join(cur,'Videos')   #enter the name whatever the name of video parent_directory is instead of the string
# cls_list=os.listdir(os.path.join(cur,'Videos'))   #enter the name whatever the name of video parent_directory is instead of the string
#
# #All output frame will be save in 'Optical Flow Frame' named folder,you can create one before executing this code
# new=os.path.join(cur,'Optical Flow Frame')
#
# try:
#     os.makedirs(new, exist_ok = True)
# except OSError as error:
#     print("Directory '%s' can not be created")
#
# os.chdir(new)
# try:
#     os.makedirs('u', exist_ok = True)
#     os.makedirs('v', exist_ok = True)
#
# except OSError as error:
#     print("Directory '%s' can not be created")
#
# dir_u=os.path.join(new,'u')
# dir_v=os.path.join(new,'v')
#
# for i in range(len(cls_list)):
#     print(cls_list[i])
#
#
#     path_u=os.path.join(dir_u,cls_list[i])
#     path_v=os.path.join(dir_v,cls_list[i])
#
#     try:
#         os.makedirs(path_u, exist_ok = True)
#         os.makedirs(path_v, exist_ok = True)
#     except OSError as error:
#         print("Directory '%s' can not be created")
#
#     sub_cls_dir=os.path.join(cls_list_dir,cls_list[i])
#     sub_cls_list=os.listdir(sub_cls_dir)  #1_21_chat .....
#     video=video_from_dir(sub_cls_dir)   ##1_21_chat
#     for xy in range(len(video)):
#         print(video[xy])
#         n=video[xy]
#         subpath_u=os.path.join(path_u,n.replace('.MP4',''))
#         subpath_v=os.path.join(path_v,n.replace('.MP4',''))
#
#         try:
#             os.makedirs(subpath_u, exist_ok = True)
#             os.makedirs(subpath_v, exist_ok = True)
#
#         except OSError as error:
#             print("Directory '%s' can not be created")
#         cap = cv2.VideoCapture(os.path.join(sub_cls_dir,video[xy]))
#         ret, frame1 = cap.read()
#         prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#         hsvx = np.zeros_like(frame1)
#         hsvx[...,1] = 10
#         hsvx[...,0] =10
#         hsvy = np.zeros_like(frame1)
#         hsvy[...,1] = 10
#         hsvy[...,0]=10
#
#         ii=0
#         while(ret):
#             ret, frame2 = cap.read()
#             ii+=1
#             if(ret):
#                 next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#             else:
#                 break
#             optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
#             flow = optical_flow.calc(prvs, next, None)
#
#             #TO CALCULATE FLOW USING FARENBACK ALGORITHM
#             #flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.8, 3, 10, 3, 7, 1.1, 0)
#
#             hsvx[...,2] = cv2.normalize(flow[...,0],None,0,255,cv2.NORM_MINMAX)
#             rgbx = cv2.cvtColor(hsvx,cv2.COLOR_HSV2BGR)
#             hsvy[...,2] = cv2.normalize(flow[...,1],None,0,255,cv2.NORM_MINMAX)
#             rgby = cv2.cvtColor(hsvy,cv2.COLOR_HSV2BGR)
#
#             cv2.imwrite(os.path.join(subpath_v , 'frame0000'+str(ii)+'.jpg'), rgby)
#             cv2.imwrite(os.path.join(subpath_u , 'frame0000'+str(ii)+'.jpg'), rgbx)
#             k = cv2.waitKey(0) & 0xff
#             prvs = next
#         cap.release()
#         cv2.destroyAllWindows()