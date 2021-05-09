from torch.autograd import Variable
import torch
from collections import deque
import numpy as np
import random
import pdb
import cv2
import sys
from torch import nn
import os
sys.path.append("game/")

GAME = 'bird'  # the name of the game being played for log files
GAMMA = 0.99  # decay rate of past observations
# 每1000个timestep，训练一次DQN网络，
OBSERVE = 1000.
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # starting value of epsilon
# 以前要记住的transitions的数量, 保留的最大的buffer记忆的数量
REPLAY_MEMORY = 50000
BATCH_SIZE = 32  # size of minibatch
FRAME_PER_ACTION = 1
UPDATE_TIME = 100
# 游戏观察的画面进行resize，特征的大小
width = 80
height = 80


def preprocess(observation):
    """
    游戏观察画面进行resize，和灰度处理, 返回处理后的观察到的状态
    :param observation: 观察的游戏画面状态 (400, 800, 3)
    :type observation:
    :return:(1, 80, 80)
    :rtype:
    """
    # 游戏观察画面进行resize，和灰度处理 --> (80,80)
    observation = cv2.cvtColor(cv2.resize(
        observation, (width, height)), cv2.COLOR_BGR2GRAY)
    # 加一个batch_size维度， (80,80) --> (1, 80, 80)
    new_observation = np.reshape(observation, (1, height, width))
    return new_observation


class DeepNetWork(nn.Module):
    def __init__(self,):
        """
        三个卷积+全连接
        """
        super(DeepNetWork,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600,256),
            nn.ReLU()
        )
        self.out = nn.Linear(256,2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        logits = self.out(x)
        return logits
class BrainDQNMain(object):
    def __init__(self, actions):
        """
        :param actions: 动作数，16
        :type actions: int
        """
        # 初始化一个记忆重放
        self.replayMemory = deque()  # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # Q网络
        self.Q_net = DeepNetWork()
        # Q的target网络
        self.Q_netT = DeepNetWork()
        self.load()
        self.loss_func = nn.MSELoss()
        LR = 1e-6
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)
    def save(self):
        print("保存模型")
        torch.save(self.Q_net.state_dict(), 'params3.pth')

    def load(self):
        """
        如果已存在，就加载已存在的模型，继续训练
        :return:
        :rtype:
        """
        if os.path.exists("params3.pth"):
            print("发现已经存在模型参数，加载模型")
            self.Q_net.load_state_dict(torch.load('params3.pth'))
            self.Q_netT.load_state_dict(torch.load('params3.pth'))

    def train(self):  # Step 1: obtain random minibatch from replay memory
        """
        训练DQN网络
        :return:
        :rtype:
        """
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3]
                           for data in minibatch]  # Step 2: calculate y
        y_batch = np.zeros([BATCH_SIZE, 1])
        # print("train next state shape")
        nextState_batch = np.array(nextState_batch)
        # print(nextState_batch.shape)
        nextState_batch = torch.Tensor(nextState_batch)
        action_batch = np.array(action_batch)
        index = action_batch.argmax(axis=1)
        print("action "+str(index))
        index = np.reshape(index, [BATCH_SIZE, 1])
        action_batch_tensor = torch.LongTensor(index)
        QValue_batch = self.Q_netT(nextState_batch)
        QValue_batch = QValue_batch.detach().numpy()

        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0] = reward_batch[i]
            else:
                # 这里的QValue_batch[i]为数组，大小为所有动作集合大小，QValue_batch[i],代表
                # 做所有动作的Q值数组，y计算为如果游戏停止，y=rewaerd[i],如果没停止，则y=reward[i]+gamma*np.max(Qvalue[i])
                # 代表当前y值为当前reward+未来预期最大值*gamma(gamma:经验系数)
                y_batch[i][0] = reward_batch[i] + \
                    GAMMA * np.max(QValue_batch[i])

        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, [BATCH_SIZE, 1])
        state_batch_tensor = Variable(torch.Tensor(state_batch))
        y_batch_tensor = Variable(torch.Tensor(y_batch))
        y_predict = self.Q_net(state_batch_tensor).gather(
            1, action_batch_tensor)
        loss = self.loss_func(y_predict, y_batch_tensor)
        print("loss is "+str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.timeStep % UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())
            self.save()

    # print(nextObservation.shape)
    def setPerception(self, nextObservation, action, reward, terminal):
        """

        :param nextObservation: 观察到的游戏状态处理后的array
        :type nextObservation:  ndarray (1, 80, 80)
        :param action: Qnet或随机的动作
        :type action: int
        :param reward: 返回的奖励
        :type reward: int
        :param terminal: 游戏是否结束
        :type terminal: int
        :return:
        :rtype:
        """
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        # 更新当前的状态， 只更新第4个层，self.currentState[1:, :, :]表示丢弃第一个层，然后append后，又变成 (4,80,80)
        newState = np.append(
            self.currentState[1:, :, :], nextObservation, axis=0)
        #加到重放缓存中， 上一个状态self.currentState:(4,80,80), 进行的动作：action， 返回的奖励：reward，返回的状态： newState: (4,80,80)。判断游戏是否结束: terminal
        self.replayMemory.append(
            (self.currentState, action, reward, newState, terminal))
        # 如果大于最大记住的buffer的数量，弹出最旧的那个
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        # 是否该训练DQN了
        if self.timeStep > OBSERVE:  # Train the network
            self.train()

        #打印一些日志信息
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", self.timeStep, "/ STATE",
              state, "/ EPSILON", self.epsilon)

        #更新上一个状态self.currentState到当前的状态newState
        self.currentState = newState
        #时间步+1
        self.timeStep += 1

    def getAction(self):
        """
        # 获取一个动作
        :return:
        :rtype:
        """
        # shape： torch.Size([1, 4, 80, 80])
        currentState = torch.Tensor([self.currentState])
        QValue = self.Q_net(currentState)[0]
        action = np.zeros(self.actions)
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                print("选择随机的动作" + str(action_index))
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue.detach().numpy())
                print("选择QNet输出的动作" + str(action_index))
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # 调整epsilon，随着时间步的增大
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        do_action = np.argmax(action)
        return do_action

    def setInitState(self, observation):
        """
        第一个状态
        :param observation: 二值化后的观测结果， shape: (80, 80)
        :type observation: array
        :return:
        :rtype:
        """
        # 形状变成--> (4, 80, 80)
        self.currentState = np.stack(
            (observation, observation, observation, observation), axis=0)
        print(f"初始化的堆叠二值化后的观测状态后的形状是: {self.currentState.shape}")


if __name__ == '__main__':
    from State import AI_Board
    game = AI_Board()
    # 可操作的动作数量
    actions = game.action_num
    # 初始化DQN
    brain = BrainDQNMain(actions)
    # 初始状态的动作，reward，和观测
    action0 = 0
    # observation0: shape: (400, 800, 3), (width, height, RGB)
    observation0, _, reward0, terminal = game.next(action0)
    # 变成灰度图，尺寸变成 (400, 800, 3) --> (80, 80)
    observation0 = cv2.cvtColor(cv2.resize(
        observation0, (width, height)), cv2.COLOR_BGR2GRAY)
    #二值化, 小于1的设为0，大于1的设为255，相当于能看到了整个图像的轮廓
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    # 设置观察到的第一个游戏状态
    brain.setInitState(observation0)
    # 开始游戏和训练
    while True:
        # 通过强化学习获取动作,获取输出的动作，action: int
        action = brain.getAction()
        #根据下一个动作，返回下一个状态和奖励
        nextObservation, _, reward, terminal = game.next(action)
        # 对观察到的游戏画面进行处理，nextObservation: shape: (1, 80, 80)
        nextObservation = preprocess(nextObservation)
        #action: int, reward: int, terminal: bool
        print(f"当前选择的动作是{action}, 返回的奖励是{reward}, 是否游戏已经结束 {terminal}")
        brain.setPerception(nextObservation, action, reward, terminal)
