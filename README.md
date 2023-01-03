velocity detection using ConvLSTM for Gelflow 1.0
fish used

1、环境配置：

如果有anaconda可以先创建一个环境 python 3.8

requirement:
numpy                         1.20.3
torch                         1.10.0+cu113
scikit-learn                  1.0.1
matplotlib                    3.4.1

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch
conda install matplotlib scikit-learn 
pip install opencv-python

安装pytorch 1.10.0，根据CUDA Version选择对应版本
在https://pytorch.org/找到对应版本，复制指令安装即可

2、数据集

SlipDataset.py中需修改数据路径

txt_path = '/home/zcf/Documents/PyCharm project/gelstereo-slip-detection/dataset/' + phase + '.txt'  # path to train.txt or test.txt
data_path = '/media/zcf/T7/slip-dataset_2.1/diff_motion'  # data path

data_path 为 相邻两帧之间marker position的差值，比如 2.npy表示marker_position/2.npy - marker_position/1.npy
get_diff_motion.py 为产生diff_motion的程序

3、训练模型 train.py

--epochs 训练轮次
--test 每训练一轮进行测试
--use_cuda 使用GPU
--checkpoint 训练结果保存在该文件夹
--model_arch 选择模型
--resume 从该模型继续训练

例如：
python train.py --epochs=50 --lr=0.01 --test --use_cuda --checkpoint=results/checkpoint_8 --model_arch=Conv
python train.py --epochs=50 --lr=0.01 --test --checkpoint=results/checkpoint_2 --model_arch=EasyLSTM
python train.py --epochs=50 --lr=0.01 --batchSize=16 --use_cuda  --test  --checkpoint=results/checkpoint_8 --model_arch=ConvLSTM
4、测试模型 test.py

--use_cuda 使用GPU
--checkpoint 被测试模型所在文件夹
--model_arch 选择模型
--test_model 被测试模型
--dataset 测试数据集 testset/trainset

例如：
python test.py --use_cuda --dataset=testset --checkpoint=results/checkpoint_8 --model_arch=Conv --test_model=model_best.pth.tar
 
python test.py --use_cuda --checkpoint=results/checkpoint_1 --model_arch=ConvLSTM


