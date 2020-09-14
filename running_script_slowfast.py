import os
import sys
print(sys.executable)



# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Sit-Stand-training-fold1-64.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Sit-Stand-testing-fold1-64.npy \
#           --result_path I:\Project\\results-Parkinson-Sit-Stand-fold1-64 \
#           --resume_path I:\\results-slowfast\\results-Parkinson-Sit-Stand-fold1-64\model_31_checkpoint.pth \
#           --model slowfastnet \
#           --n_classes 13 \
#           --n_epoch 51 \
#           --sample_size 112 \
#           --weight_decay 0.00001 \
#           --lr_steps 50 \
#           --learning_rate 0.01 \
#           --batch_size 16 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Sit-Stand-training-fold2-64.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Sit-Stand-testing-fold2-64.npy \
#           --result_path I:\Project\\results-Parkinson-Sit-Stand-fold2-64 \
#           --resume_path I:\\results-slowfast\\results-Parkinson-Sit-Stand-fold2-64\model_31_checkpoint.pth \
#           --model slowfastnet \
#           --n_classes 13 \
#           --n_epoch 51 \
#           --sample_size 112 \
#           --weight_decay 0.00001 \
#           --lr_steps 50 \
#           --learning_rate 0.01 \
#           --batch_size 16 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Sit-Stand-training-fold3-64.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Sit-Stand-testing-fold3-64.npy \
#           --result_path I:\Project\\results-Parkinson-Sit-Stand-fold3-64 \
#           --resume_path I:\\results-slowfast\\results-Parkinson-Sit-Stand-fold3-64\model_31_checkpoint.pth \
#           --model slowfastnet \
#           --n_classes 13 \
#           --n_epoch 51 \
#           --sample_size 112 \
#           --weight_decay 0.00001 \
#           --learning_rate 0.01 \
#           --lr_steps 50 \
#           --batch_size 16 \
#           --n_threads 8 ''')
# #
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Sit-Stand-training-fold4-64.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Sit-Stand-testing-fold4-64.npy \
#           --result_path I:\Project\\results-Parkinson-Sit-Stand-fold4-64 \
#           --resume_path I:\\results-slowfast\\results-Parkinson-Sit-Stand-fold4-64\model_31_checkpoint.pth \
#           --model slowfastnet \
#           --n_classes 13 \
#           --n_epoch 51 \
#           --sample_size 112 \
#           --weight_decay 0.00001 \
#           --learning_rate 0.01 \
#           --lr_steps 50 \
#           --batch_size 16 \
#           --n_threads 8 ''')

# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Walk-training-fold1-64.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Walk-testing-fold1-64.npy \
#           --result_path I:\Project\\results-Parkinson-Walk-fold1-64 \
#           --resume_path I:\\results-slowfast\\results-Parkinson-Walk-fold1-64\model_31_checkpoint.pth \
#           --model slowfastnet \
#           --n_classes 5 \
#           --n_epoch 51 \
#           --sample_size 112 \
#           --weight_decay 0.00001 \
#           --learning_rate 0.01 \
#           --lr_steps 50 \
#           --batch_size 16 \
#           --n_threads 8 ''')
#
os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Walk-training-fold2-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Walk-testing-fold2-64.npy \
          --result_path I:\Project\\results-Parkinson-Walk-fold2-64 \
          --model slowfastnet \
          --n_classes 5 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Walk-training-fold3-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Walk-testing-fold3-64.npy \
          --result_path I:\Project\\results-Parkinson-Walk-fold3-64 \
          --resume_path I:\\results-slowfast\\results-Parkinson-Walk-fold3-64\model_31_checkpoint.pth \
          --model slowfastnet \
          --n_classes 5 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Walk-training-fold4-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Parkinson-Walk-testing-fold4-64.npy \
          --result_path I:\Project\\results-Parkinson-Walk-fold4-64 \
          --resume_path I:\\results-slowfast\\results-Parkinson-Walk-fold4-64\model_31_checkpoint.pth \
          --model slowfastnet \
          --n_classes 5 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')


os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Walk-training-fold1-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Walk-testing-fold1-64.npy \
          --result_path I:\Project\\results-Stroke-Walk-fold1-64 \
          --resume_path I:\\results-slowfast\\results-Stroke-Walk-fold1-64\model_31_checkpoint.pth \
          --model slowfastnet \
          --n_classes 6 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Walk-training-fold2-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Walk-testing-fold2-64.npy \
          --result_path I:\Project\\results-Stroke-Walk-fold2-64 \
          --resume_path I:\\results-slowfast\\results-Stroke-Walk-fold2-64\model_31_checkpoint.pth \
          --model slowfastnet \
          --n_classes 6 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Walk-training-fold3-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Walk-testing-fold3-64.npy \
          --result_path I:\Project\\results-Stroke-Walk-fold3-64 \
          --resume_path I:\\results-slowfast\\results-Stroke-Walk-fold3-64\model_31_checkpoint.pth \
          --model slowfastnet \
          --n_classes 6 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Walk-training-fold4-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Walk-testing-fold4-64.npy \
          --result_path I:\Project\\results-Stroke-Walk-fold4-64 \
          --resume_path I:\\results-slowfast\\results-Stroke-Walk-fold4-64\model_31_checkpoint.pth \
          --model slowfastnet \
          --n_classes 6 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Sit-Stand-training-fold1-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Sit-Stand-testing-fold1-64.npy \
          --result_path I:\Project\\results-Stroke-Sit-Stand-fold1-64 \
          --resume_path I:\\results-slowfast\\results-Stroke-Sit-Stand-fold1-64\model_31_checkpoint.pth \
          --model slowfastnet \
          --n_classes 6 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Sit-Stand-training-fold2-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Sit-Stand-testing-fold2-64.npy \
          --result_path I:\Project\\results-Stroke-Sit-Stand-fold2-64 \
          --resume_path I:\\results-slowfast\\results-Stroke-Sit-Stand-fold2-64\model_31_checkpoint.pth \
          --model slowfastnet \
          --n_classes 6 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Sit-Stand-training-fold3-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Sit-Stand-testing-fold3-64.npy \
          --result_path I:\Project\\results-Stroke-Sit-Stand-fold3-64 \
          --resume_path I:\\results-slowfast\\results-Stroke-Sit-Stand-fold3-64\model_31_checkpoint.pth \
          --model slowfastnet \
          --n_classes 6 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Sit-Stand-training-fold4-64.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\\64\Stroke-Sit-Stand-testing-fold4-64.npy \
          --result_path I:\Project\\results-Stroke-Sit-Stand-fold4-64 \
          --resume_path I:\\results-slowfast\\results-Stroke-Sit-Stand-fold4-64\model_31_checkpoint.pth \
          --model slowfastnet \
          --n_classes 6 \
          --n_epoch 51 \
          --sample_size 112 \
          --weight_decay 0.00001 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --batch_size 16 \
          --n_threads 8 ''')