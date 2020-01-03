# MachineLearningFinalProj
A Kaggle project about the classification of 3D Voxel of lung modules

运行test.py可以利用DenseNet_parameter3.pkl在ykl_sub_test.csv文件中生成Submission文件；
DenseNet.py是网络的设计；
mydata.py是对archive中的数据进行处理，只需要将数据作为一个整体archive加入到仓库解压后的文件内即可以运行；
main_py.py和main_py2.py主程序框架大致相同，仅是为了训练时比较不同的技巧和参数时设置；
最终的训练以采用mixup+transform为好。
