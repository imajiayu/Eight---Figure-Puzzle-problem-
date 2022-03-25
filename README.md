# A星搜索求解八数码问题

### 1950509 马家昱

![avatar](./screenshots.png)

## 问题描述

在九宫格中放入0~8共九个数字，0表示空格，其余数字可以与空格交换位置。给定初始状态与目标状态，求解中间步骤与需要的最小步数。

![avatar](./1.png)

## 求解思路：
### 问题的可解性
并非所有八数码问题均有有限步数的可行解。若将九宫格转变为线性结构，可以证明，在将其中某一数字与空位置交换位置后，其逆序数的奇偶性不变。因此，计算其初始状态与目标状态的逆序数，即可判断该问题有无解。

### A星算法的基本思路

Dijkstra算法遍历整个解空间以获得最优解，但其时间代价过于庞大。而最佳优选搜索只考虑当前节点与目标节点的距离，所得到的解不一定是最优解。因此，将这两种算法结合起来，得到下面的公式：

$F(n)=G(n)+H(n)$

$G(n)$表示当前节点到初始节点的距离，$H(n)$表示当前节点到目标节点的距离。当$G(n)$为0时，变为最佳优选搜索；当$H(n)$为0时，变为Dijkstra算法。

其基本步骤如下：
* 维护两张表open_list与close_list,前者表示待搜索的节点，后者表示已经搜索过的节点。
* 将初始状态放入open_list中，遍历其产生的新的节点，计算这些新节点的G、H、F值，并且记录其父节点为初始节点，将其加入open_list。将初始节点从open_list取出放入close_list。
* 在open_list中选择F值最小的节点，重复第二步的过程，直到找到目标状态。
* 在搜索过程中，若搜索到的节点已经存在于open_list，则比较这两个节点的G值。若新节点的G值更小，说明由当前状态到达这一状态的代价更小，因此更新其G值与父节点，否则什么也不做。

### 在八数码问题中使用A星算法

考虑$G(n)$与$H(n)$的计算方法
* $G(n)$表示与初始节点的距离。因此，设初始节点的G为0，则之后搜索到的每一个节点的G值均为其父节点的G值+1。
* $H(n)$表示与目标节点的距离。首先尝试使用当前节点与目标节点对应位置数字不同的个数为$H(n)$值，发现收敛速度过慢。改进后使用对应数字的距离之和作为$H(n)$值。如图所示,两个"1"之间的距离为2，而"0"之间的距离为3。

![avatar](./1.png)

## 实现代码
```py
import wx
import typing
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import time

start=None
S_end=np.array([[1,2,3],[8,0,4],[7,6,5]])

def random_init_status():  # 随机产生状态，并用numpy.array存储
    seq = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    random.shuffle(seq)
    result = np.resize(np.array(seq), (3, 3))
    return result


def show_status(S, steps):  # 展示九宫格状态
    plt.imshow(S)
    plt.axis('off')
    
    for i in range(0, 3):
        for j in range(0, 3):
            plt.text(x=i-0.06, y=j+0.07, s=S[j][i], fontsize=20)  # 打印矩阵对应位置的值
    if steps>=0:
        plt.text(x=0.65, y=0.5, s="steps:{}".format(steps), fontsize=20)  # 打印当前步数

    # plt.savefig("./{}.jpg".format(steps))
    plt.pause(0.5)
    plt.cla()


def isEqual(a, b):  # 判断两个状态是否相等
    return (a == b).all()


def get_zero_arg(S):  # 获取空位置的下标
    return np.argwhere(S == 0)[0]


def get_h(S1, S2):  # 计算当前状态的H(x)值，使用的是两个状态间的距离，即每个位置的距离之和
    count = 0
    for i in range(0, 3):
        for j in range(0, 3):
            target_loc = np.argwhere(S2 == S1[i][j])[0]
            count += (abs(target_loc[0]-i)+abs(target_loc[1]-j))

    return count


def inverse_number_parity(S1, S2):  # 求两个状态的逆序数，以此判断该问题可不可解
    s1 = np.resize(S1, (1, 9))
    s1_cnt = 0
    for i in range(len(s1[0])):
        if s1[0][i] == 0:
            continue
        for j in range(i):
            if s1[0][j] > s1[0][i]:
                s1_cnt += 1

    s2 = np.resize(S2, (1, 9))
    s2_cnt = 0
    for i in range(len(s2[0])):
        if s2[0][i] == 0:
            continue
        for j in range(i):
            if s2[0][j] > s2[0][i]:
                s2_cnt += 1

    if s1_cnt % 2 == s2_cnt % 2:
        return True
    else:
        return False


def inboard(newX, newY):  # 位置是否在九宫格内
    if newX >= 0 and newX <= 2 and newY >= 0 and newY <= 2:
        return True
    else:
        return False


def show_steps(close_list, node):  # 递归展示当前状态，以此演示从初始状态到目标状态的步骤
    if node[1] is None:
        show_status(node[0], node[2])
        return
    for s in close_list:
        if isEqual(s[0], node[0]):
            for t in close_list:
                if isEqual(s[1], t[0]):
                    show_steps(close_list, t)
                    show_status(s[0], node[2])
                    return


def show_start_end(S_start, S_end):  # 展示初始和目标状态
    plt.imshow(S_start)
    plt.axis('off')
    for i in range(0, 3):
        for j in range(0, 3):
            plt.text(x=i-0.06, y=j+0.07, s=S_start[j][i], fontsize=20)
    plt.text(x=0.7, y=0.5, s="Begin", fontsize=20)
    plt.pause(1)
    plt.cla()

    plt.imshow(S_end)
    plt.axis('off')
    for i in range(0, 3):
        for j in range(0, 3):
            plt.text(x=i-0.06, y=j+0.07, s=S_end[j][i], fontsize=20)
    plt.text(x=0.8, y=0.5, s="End", fontsize=20)
    plt.pause(1)
    plt.cla()

def Astar(S_start, S_end):  # A星算法

    open_list = []
    close_list = []

    # 将初始状态加入open_list,其父状态为None,后三个元素分别为G、H、F
    open_list.append([S_start, None, 0, 0, 0])
    time_start = time.time()
    while(True):  # 主循环
        s = open_list[0]  # 取F值最小的节点

        open_list.pop(0)
        close_list.append(s)  # 将其从open_list中移除，加入close_list

        if isEqual(s[0], S_end):  # 找到目标状态
            time_end = time.time()
            return close_list,s,time_end-time_start

        dirX = [-1, 1, 0, 0]
        dirY = [0, 0, -1, 1]  # 以空位为中心，上下左右搜索的四个方向

        zero_loc = get_zero_arg(s[0])  # 空位的下标
        for i in range(0, 4):  # 搜索的四个方向
            newX = zero_loc[0]+dirX[i]
            newY = zero_loc[1]+dirY[i]
            if inboard(newX, newY):  # 在九宫格内
                new_s = copy.deepcopy(s[0])
                new_s[zero_loc[0]][zero_loc[1]
                                   ], new_s[newX][newY] = new_s[newX][newY], new_s[zero_loc[0]][zero_loc[1]]  # 产生新状态
                G = s[2]+1  # G为已经走过的步数，为父节点走过的步数+1
                H = get_h(new_s, S_end)  # 获取H值
                F = G+H

                already_in_list = 0

                for open_s in open_list:  # 若新状态已经在open_list中，则比较G值
                    if isEqual(new_s, open_s[0]):
                        if G < open_s[2]:
                            open_s[2] = G
                            open_s[1] = s
                            open_s[4] = open_s[2]+open_s[3]
                            already_in_list = 1
                            break

                for close_s in close_list:  # 若新状态已经在close_list中，什么都不做
                    if isEqual(new_s, close_s[0]):
                        already_in_list = 1
                        break

                if not already_in_list:  # 否则加入open_list
                    open_list.append([new_s, s[0], G, H, F])

        open_list.sort(key=lambda x: x[4])  # 按照F值排序，使open_list第一个节点的F值最小

###可视化部分
app = wx.App()
window = wx.Frame(None, title="八数码问题", size=(300, 180))
panel = wx.Panel(window)

cus_list=[1, 2, 3, 4, 5, 6, 7, 8, 0]
is_slovable=0
is_playable=0
close_list_s=None

def judge_cus(event):
    global is_slovable
    global start
    is_slovable=0
    if len(set(cus_list)) != 9:
        toastone = wx.MessageDialog(
            None, "9个数字不能重复！", "错误输入", wx.YES_DEFAULT | wx.ICON_QUESTION)
        if toastone.ShowModal() == wx.ID_YES:  # 如果点击了提示框的确定按钮
            toastone.Destroy()
        return
    S_start=np.resize(np.array(cus_list),(3,3))
    if not inverse_number_parity(S_start,S_end):
        toastone = wx.MessageDialog(
            None, "该问题不可解！", "错误输入", wx.YES_DEFAULT | wx.ICON_QUESTION)
        if toastone.ShowModal() == wx.ID_YES:  # 如果点击了提示框的确定按钮
            toastone.Destroy()
        return
    toastone = wx.MessageDialog(
            None, "生成初始状态成功！", "提示", wx.YES_DEFAULT | wx.ICON_QUESTION)
    if toastone.ShowModal() == wx.ID_YES:  # 如果点击了提示框的确定按钮
        toastone.Destroy()
    is_slovable=1
    start=S_start

def judge_rand(event):
    global is_slovable
    global start
    while(1):
        S_start=random_init_status()
        if inverse_number_parity(S_start,S_end):
            start=S_start
            break
    toastone = wx.MessageDialog(
            None, "生成初始状态成功！", "提示", wx.YES_DEFAULT | wx.ICON_QUESTION)
    if toastone.ShowModal() == wx.ID_YES:  # 如果点击了提示框的确定按钮
        toastone.Destroy()
    is_slovable=1

def cal(event):
    global is_slovable
    global is_playable
    global close_list_s
    is_playable=0
    if is_slovable==0:
        toastone = wx.MessageDialog(
            None, "尚未生成初始状态！", "提示", wx.YES_DEFAULT | wx.ICON_QUESTION)
        if toastone.ShowModal() == wx.ID_YES:  # 如果点击了提示框的确定按钮
            toastone.Destroy()
        return
    result=Astar(start,S_end)
    close_list_s=result
    toastone = wx.MessageDialog(
            None, "计算完毕！\n 计算时间为：{0} \n 步数：{1}".format(result[2],result[1][2]), "提示", wx.YES_DEFAULT | wx.ICON_QUESTION)
    if toastone.ShowModal() == wx.ID_YES:  # 如果点击了提示框的确定按钮
        toastone.Destroy()
    is_playable=1

def run(event):
    global close_list_s
    global is_playable
    global start
    if is_playable==0:
        toastone = wx.MessageDialog(
            None, "尚未计算！", "提示", wx.YES_DEFAULT | wx.ICON_QUESTION)
        if toastone.ShowModal() == wx.ID_YES:  # 如果点击了提示框的确定按钮
            toastone.Destroy()
        return
    show_start_end(start,S_end)
    show_steps(close_list_s[0],close_list_s[1])
     
cus_start_btn = wx.Button(panel, -1, "自定义初始状态")
cus_start_btn.Bind(wx.EVT_BUTTON, judge_cus)

random_start_btn = wx.Button(panel, -1, "随机生成初始状态", pos=(140, 0))
random_start_btn.Bind(wx.EVT_BUTTON, judge_rand)

cal_start_btn = wx.Button(panel, -1, "开始计算", pos=(140, 30))
cal_start_btn.Bind(wx.EVT_BUTTON, cal)

play_start_btn = wx.Button(panel, -1, "开始演示", pos=(140, 60))
play_start_btn.Bind(wx.EVT_BUTTON,run)


choices_list = [str(x) for x in range(0, 9)]
ch1 = wx.ComboBox(panel, -1, value='1', choices=choices_list,
                  style=wx.CB_SORT, pos=(0, 30))
ch2 = wx.ComboBox(panel, -1, value='2', choices=choices_list,
                  style=wx.CB_SORT, pos=(40, 30))
ch3 = wx.ComboBox(panel, -1, value='3', choices=choices_list,
                  style=wx.CB_SORT, pos=(80, 30))
ch4 = wx.ComboBox(panel, -1, value='4', choices=choices_list,
                  style=wx.CB_SORT, pos=(0, 60))
ch5 = wx.ComboBox(panel, -1, value='5', choices=choices_list,
                  style=wx.CB_SORT, pos=(40, 60))
ch6 = wx.ComboBox(panel, -1, value='6', choices=choices_list,
                  style=wx.CB_SORT, pos=(80, 60))
ch7 = wx.ComboBox(panel, -1, value='7', choices=choices_list,
                  style=wx.CB_SORT, pos=(0, 90))
ch8 = wx.ComboBox(panel, -1, value='8', choices=choices_list,
                  style=wx.CB_SORT, pos=(40, 90))
ch9 = wx.ComboBox(panel, -1, value='0', choices=choices_list,
                  style=wx.CB_SORT, pos=(80, 90))


def get11(event):
    cus_list[0] = int(event.GetString())


def get12(event):
    cus_list[1] = int(event.GetString())


def get13(event):
    cus_list[2] = int(event.GetString())


def get21(event):
    cus_list[3] = int(event.GetString())


def get22(event):
    cus_list[4] = int(event.GetString())


def get23(event):
    cus_list[5] = int(event.GetString())


def get31(event):
    cus_list[6] = int(event.GetString())


def get32(event):
    cus_list[7] = int(event.GetString())


def get33(event):
    cus_list[8] = int(event.GetString())


ch1.Bind(wx.EVT_COMBOBOX, get11)
ch2.Bind(wx.EVT_COMBOBOX, get12)
ch3.Bind(wx.EVT_COMBOBOX, get13)
ch4.Bind(wx.EVT_COMBOBOX, get21)
ch5.Bind(wx.EVT_COMBOBOX, get22)
ch6.Bind(wx.EVT_COMBOBOX, get23)
ch7.Bind(wx.EVT_COMBOBOX, get31)
ch8.Bind(wx.EVT_COMBOBOX, get32)
ch9.Bind(wx.EVT_COMBOBOX, get33)


window.Show(True)
app.MainLoop()

```

## 实验结果

使用wxpython首先GUI界面，效果如下：

![avatar](./3.png)

* 自定义或随机初始化状态

![avatar](./4.png)

存在错误输入检测

![avatar](./7.png)

* 开始计算，输出时间和步数

![avatar](./5.png)

* 动态演示

![avatar](./6.png)
  
某一实例的搜索过程如下：

![avatar](./2.png)
