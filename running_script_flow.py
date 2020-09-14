import os
import sys
print(sys.executable)


# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow-tvl1\QMAR-Dataset\Parkinson-Sit-Stand-training-flow-fold1.npy \
#           --testset_path_flow I:\Optical-Flow-tvl1\QMAR-Dataset\Parkinson-Sit-Stand-testing-flow-fold1.npy \
#           --result_path I:\Project\\results-flow-tvl1\\results-Parkinson-Sit-Stand-fold1 \
#           --pretrain_path I:\Project\pretrained\r3d101_K_200ep.pth \
#           --resume_path I:\results-flow-tvl1-resnet101-0.01\results-Parkinson-Sit-Stand-fold1\model_31_checkpoint.pth \
#           --model flow \
#           --model_depth 101 \
#           --n_classes 13 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.01 \
#           --lr_steps 50 \
#           --n_epochs 51 \
#           --batch_size 5 \
#           --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path_flow I:\Optical-Flow-tvl1\QMAR-Dataset\Parkinson-Sit-Stand-training-flow-fold2.npy \
          --testset_path_flow I:\Optical-Flow-tvl1\QMAR-Dataset\Parkinson-Sit-Stand-testing-flow-fold2.npy \
          --result_path I:\Project\\results-flow-tvl1\\results-Parkinson-Sit-Stand-fold2 \
          --resume_path I:\\results-flow-tvl1-resnet101-0.01\\results-Parkinson-Sit-Stand-fold2\model_31_checkpoint.pth \
          --model flow \
          --model_depth 101 \
          --n_classes 13 \
          --sample_size 112 \
          --sample_duration 16 \
          --learning_rate 0.01 \
          --lr_steps 50 \
           --n_epochs 51 \
          --batch_size 5 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path_flow I:\Optical-Flow-tvl1\QMAR-Dataset\Parkinson-Sit-Stand-training-flow-fold3.npy \
          --testset_path_flow I:\Optical-Flow-tvl1\QMAR-Dataset\Parkinson-Sit-Stand-testing-flow-fold3.npy \
          --result_path I:\Project\\results-flow-tvl1\\results-Parkinson-Sit-Stand-fold3 \
          --resume_path I:\\results-flow-tvl1-resnet101-0.01\\results-Parkinson-Sit-Stand-fold3\model_31_checkpoint.pth \
          --model flow \
          --model_depth 101 \
          --n_classes 13 \
          --sample_size 112 \
          --sample_duration 16 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --n_epochs 51 \
          --batch_size 5 \
          --n_threads 8 ''')

os.system('''python main.py \
          --annotation_path_flow I:\Optical-Flow-tvl1\QMAR-Dataset\Parkinson-Sit-Stand-training-flow-fold4.npy \
          --testset_path_flow I:\Optical-Flow-tvl1\QMAR-Dataset\Parkinson-Sit-Stand-testing-flow-fold4.npy \
          --result_path I:\Project\\results-flow-tvl1\\results-Parkinson-Sit-Stand-fold4 \
          --resume_path I:\\results-flow-tvl1-resnet101-0.01\\results-Parkinson-Sit-Stand-fold4\model_31_checkpoint.pth \
          --model flow \
          --model_depth 101 \
          --n_classes 13 \
          --sample_size 112 \
          --sample_duration 16 \
          --learning_rate 0.01 \
          --lr_steps 50 \
          --n_epochs 51 \
          --batch_size 5 \
          --n_threads 8 ''')

# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Sit-Stand-training-flow-fold2.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Sit-Stand-testing-flow-fold2.npy \
#           --result_path I:\Project\\results-flow\\results-Parkinson-Sit-Stand-fold2 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 13 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Sit-Stand-training-flow-fold3.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Sit-Stand-testing-flow-fold3.npy \
#           --result_path I:\Project\\results-flow\\results-Parkinson-Sit-Stand-fold3 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 13 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Walk-training-flow-fold1.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Walk-testing-flow-fold1.npy \
#           --result_path I:\Project\\results-flow\\results-Parkinson-Walk-fold1 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 5 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Walk-training-flow-fold2.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Walk-testing-flow-fold2.npy \
#           --result_path I:\Project\\results-flow\\results-Parkinson-Walk-fold2 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 5 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Walk-training-flow-fold3.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Walk-testing-flow-fold3.npy \
#           --result_path I:\Project\\results-flow\\results-Parkinson-Walk-fold3 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 5 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Walk-training-flow-fold4.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Parkinson-Walk-testing-flow-fold4.npy \
#           --result_path I:\Project\\results-flow\\results-Parkinson-Walk-fold4 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 5 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow "I:\Optical-Flow\QMAR-Dataset\Stroke-Walk-training-flow-fold1.npy" \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Stroke-Walk-testing-flow-fold1.npy \
#           --result_path I:\Project\\results-flow\\results-Stroke-Walk-fold1 \
#           --pretrain_path I:\Project\pretrained\\r3d50_K_200ep.pth \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')

# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Stroke-Walk-training-flow-fold2.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Stroke-Walk-testing-flow-fold2.npy \
#           --result_path I:\Project\\results-flow\\results-Stroke-Walk-fold2 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')

# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Stroke-Walk-training-flow-fold3.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Stroke-Walk-testing-flow-fold3.npy \
#           --result_path I:\Project\\results-flow\\results-Stroke-Walk-fold3 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Sroke-Walk-training-flow-fold4.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Stroke-Walk-testing-flow-fold4.npy \
#           --result_path I:\Project\\results-flow\\results-Stroke-Walk-fold4 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Sroke-Sit-Stand-training-flow-fold1.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Stroke-Sit-Stand-testing-flow-fold1.npy \
#           --result_path I:\Project\\results-flow\\results-Stroke-Sit-Stand-fold1 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Sroke-Sit-Stand-training-flow-fold2.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Stroke-Sit-Stand-testing-flow-fold2.npy \
#           --result_path I:\Project\\results-flow\\results-Stroke-Sit-Stand-fold2 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Sroke-Sit-Stand-training-flow-fold3.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Stroke-Sit-Stand-testing-flow-fold3.npy \
#           --result_path I:\Project\\results-flow\\results-Stroke-Sit-Stand-fold3 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')
#
# os.system('''python main.py \
#           --annotation_path_flow I:\Optical-Flow\QMAR-Dataset\Sroke-Sit-Stand-training-flow-fold4.npy \
#           --testset_path_flow I:\Optical-Flow\QMAR-Dataset\Stroke-Sit-Stand-testing-flow-fold4.npy \
#           --result_path I:\Project\\results-flow\\results-Stroke-Sit-Stand-fold4 \
#           --model flow \
#           --model_depth 50 \
#           --n_classes 6 \
#           --sample_size 112 \
#           --sample_duration 16 \
#           --learning_rate 0.001 \
#           --batch_size 5 \
#           --n_threads 8 ''')


