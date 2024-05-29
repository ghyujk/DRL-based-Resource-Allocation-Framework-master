# Deep Reinforcement Learning based Resource Allocation Framework

## 这是什么？
这是一个基于深度强化学习(Deep Reinforcement Learning)的资源分配算法，它能够根据用户信道条件的好坏，动态地分配子信道和传输功率，最大化非正交多址调制(NOMA)系统的能量效率。
本项目基于Deep Q Learning Network(DQN)和Deep Deterministic Policy Gradient(DDPG)算法。

如果你是机器学习，特别是深度强化学习的新手，又正好在进行通信邻域的智能算法的研究，那么本项目是你了解强化学习算法的不二之选！当然，强化学习的优势之一就是它可以被轻易地迁移到其他应用场景之中。因此，如果你是别的邻域的研究人员，相信本项目一样对你有参考价值。

如果你想进一步了解此算法的理论原理，欢迎查阅我的论文，IEEE链接：[https://ieeexplore.ieee.org/document/8927898/](https://ieeexplore.ieee.org/document/8927898/)

## 如何开始？
首先确保你安装了Python，以及下列库文件：

numpy：[https://numpy.org/](https://numpy.org/) 用于对矩阵，列表等数据进行处理。

pandas：[https://pandas.pydata.org/](https://pandas.pydata.org/) 一样是用于对数据进行处理，特别是对csv文件进行导出和导入。

tensorflow：[https://tensorflow.google.cn/](https://tensorflow.google.cn/) 机器学习的核心库文件。

keras：[https://keras.io/zh/](https://keras.io/zh/) 一个基于Python的高级神经网络API。

克隆本项目的代码到你喜欢的任意位置。然后，你只需要打开根目录下的run.py文件，即可以轻松运行！
(Note：在Pycharm下记得选择正确的Project Interpreter)

本项目通过深度强化学习算法，得到每个时隙下，适合用户信道状态的子信道分配和功率分配方案。之后，在控制台Print出所有用户的能量效率之和作为结果。
所以，当你在控制台能够看到有"DQN_rate: xxxxx"的结果输出，说明你的安装步骤正确。

## 这有什么特别之处？

1.  基于机器学习的智能资源分配算法：归功于深度神经网络的强大计算能力，与传统启发式算法相比，本方案的计算时间大大降低。
2.  基于深度强化学习的资源分配算法：与深度学习(Deep Learning)相比，深度强化学习算法的优势在于不需要预先花费大量时间和精力训练你的神经网络。
3.  多DQN的网络架构：使用多个DQN网络进行资源的分配(如果你打开根目录下DRL这个文件夹，会发现有许多个DQNxxx.py文件)，与只使用一个网络相比，本算法可以大大降低Action的维度大小，从而提高算法的效率。

######
在深度强化学习中，通常会针对不同的问题或环境设计不同的网络结构或改进的DQN变体。这就是为什么可能会有多个不同命名的DQN文件（例如DQNxxx.py）存在于DRL文件夹中。
每个DQN文件可能包含一个不同版本或变种的深度Q网络（DQN），其结构、超参数或训练方式可能会有所不同。这种多样性可以是为了尝试不同的架构、改进、优化方法或用于解决不同类型问题的特定模型。
例如，可能存在这样的文件：
DQN_Agent.py
DQN_Double.py
DQN_Dueling.py
每个文件代表了一个不同的DQN变种，可能对应于使用不同技术改进的版本，如双重DQN、Dueling DQN等。
这种做法允许研究者或开发者可以轻松地尝试不同的网络结构或改进，以便比较它们在特定环境或问题上的性能表现，并选择最有效的模型来解决特定任务。
######

## 后续计划？
本项目后续还将继续添加DDPG等更高级的深度强化学习算法......