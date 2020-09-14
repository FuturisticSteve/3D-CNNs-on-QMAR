import os
import sys
print(sys.executable)


# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Stroke-Walk-training-fold1.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Stroke-Walk-testing-fold1.npy \
#           --result_path I:\Project\\results-Stroke-Walk-fold1 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --lr_steps 20 40 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Stroke-Walk-training-fold2.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Stroke-Walk-testing-fold2.npy \
#           --result_path I:\Project\\results-Stroke-Walk-fold2 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --lr_steps 20 40 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Stroke-Walk-training-fold3.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Stroke-Walk-testing-fold3.npy \
#           --result_path I:\Project\\results-Stroke-Walk-fold3 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --lr_steps 20 40 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Stroke-Walk-training-fold4.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Stroke-Walk-testing-fold4.npy \
#           --result_path I:\Project\\results-Stroke-Walk-fold4 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --lr_steps 20 40 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Stroke-Sit-Stand-training-fold1.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Stroke-Sit-Stand-testing-fold1.npy \
#           --result_path I:\Project\\results-Stroke-Sit-Stand-fold1 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Stroke-Sit-Stand-training-fold2.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Stroke-Sit-Stand-testing-fold2.npy \
#           --result_path I:\Project\\results-Stroke-Sit-Stand-fold2 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
# #
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Stroke-Sit-Stand-training-fold3.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Stroke-Sit-Stand-testing-fold3.npy \
#           --result_path I:\Project\\results-Stroke-Sit-Stand-fold3 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Stroke-Sit-Stand-training-fold4.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Stroke-Sit-Stand-testing-fold4.npy \
#           --result_path I:\Project\\results-Stroke-Sit-Stand-fold4 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Parkinson-Walk-training-fold1.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Parkinson-Walk-testing-fold1.npy \
#           --result_path I:\Project\\results-Parkinson-Walk-fold1 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 5 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Parkinson-Walk-training-fold2.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Parkinson-Walk-testing-fold2.npy \
#           --result_path I:\Project\\results-Parkinson-Walk-fold2 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 5 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Parkinson-Walk-training-fold3.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Parkinson-Walk-testing-fold3.npy \
#           --result_path I:\Project\\results-Parkinson-Walk-fold3 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 5 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
# os.system('''python main.py \
#           --annotation_path I:\Faegheh\QMAR-Dataset\Parkinson-Walk-training-fold4.npy \
#           --testset_path I:\Faegheh\QMAR-Dataset\Parkinson-Walk-testing-fold4.npy \
#           --result_path I:\Project\\results-Parkinson-Walk-fold4 \
#           --pretrain_path I:\Project\pretrained\\r3d101_K_200ep.pth \
#           --n_finetune_classes 700 \
#           --ft_portion complete \
#           --model resnet \
#           --model_depth 101 \
#           --width_mult 0.45 \
#           --train_crop center \
#           --n_classes 5 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 \
#           --checkpoint 1 \
#           --n_val_samples 1''')
#
os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\Parkinson-Sit-Stand-training-fold1.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\Parkinson-Sit-Stand-testing-fold1.npy \
          --result_path I:\Project\\results-Parkinson-Sit-Stand-fold1 \
          --model resnet \
          --model_depth 101 \
          --n_classes 13 \
          --sample_size 112 \
          --sample_duration 16 \
          --learning_rate 0.001 \
          --batch_size 5 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\Parkinson-Sit-Stand-training-fold2.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\Parkinson-Sit-Stand-testing-fold2.npy \
          --result_path I:\Project\\results-Parkinson-Sit-Stand-fold2 \
          --n_finetune_classes 700 \
          --ft_portion complete \
          --model resnet \
          --model_depth 101 \
          --width_mult 0.45 \
          --train_crop center \
          --n_classes 13 \
          --sample_size 112 \
          --sample_duration 16 \
          --learning_rate 0.001 \
          --batch_size 5 \
          --n_threads 8 \
          --checkpoint 1 \
          --n_val_samples 1''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\Parkinson-Sit-Stand-training-fold3.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\Parkinson-Sit-Stand-testing-fold3.npy \
          --result_path I:\Project\\results-Parkinson-Sit-Stand-fold3 \
          --n_finetune_classes 700 \
          --ft_portion complete \
          --model resnet \
          --model_depth 101 \
          --width_mult 0.45 \
          --train_crop center \
          --n_classes 13 \
          --sample_size 112 \
          --sample_duration 16 \
          --learning_rate 0.001 \
          --batch_size 5 \
          --n_threads 8 \
          --checkpoint 1 \
          --n_val_samples 1''')

os.system('''python main.py \
          --annotation_path I:\Faegheh\QMAR-Dataset\Parkinson-Sit-Stand-training-fold4.npy \
          --testset_path I:\Faegheh\QMAR-Dataset\Parkinson-Sit-Stand-testing-fold4.npy \
          --result_path I:\Project\\results-Parkinson-Sit-Stand-fold4 \
          --n_finetune_classes 700 \
          --ft_portion complete \
          --model resnet \
          --model_depth 101 \
          --width_mult 0.45 \
          --train_crop center \
          --n_classes 13 \
          --sample_size 112 \
          --sample_duration 16 \
          --learning_rate 0.001 \
          --batch_size 5 \
          --n_threads 8 \
          --checkpoint 1 \
          --n_val_samples 1''')
