# 阶段总结：手势交互中的手势识别、分类与三维运动追踪

- [阶段总结：手势交互中的手势识别、分类与三维运动追踪](#阶段总结手势交互中的手势识别分类与三维运动追踪)
  - [一、前言](#一前言)
  - [二、相机选型](#二相机选型)
  - [三、运行demo](#三运行demo)
  - [四、mediapipe手势识别](#四mediapipe手势识别)
  - [五、基于MLP和PointNet的手势分类](#五基于mlp和pointnet的手势分类)
    - [1.数据结构和类型](#1数据结构和类型)
    - [2.MLP手势分类](#2mlp手势分类)
      - [2.1数据预处理——MLP](#21数据预处理mlp)
      - [2.2网络结构——MLP](#22网络结构mlp)
      - [2.3训练网络——MLP](#23训练网络mlp)
    - [3.PointNet手势分类](#3pointnet手势分类)
      - [1.PointNet的结构](#1pointnet的结构)
      - [2.PointNet的数据预处理](#2pointnet的数据预处理)
      - [3.PointNet的训练过程](#3pointnet的训练过程)
      - [4.PointNet和MLP的对比](#4pointnet和mlp的对比)
  - [六、手势三维追踪](#六手势三维追踪)
    - [1.窗口深度搜索](#1窗口深度搜索)
    - [2.滑动加权平均滤波](#2滑动加权平均滤波)
    - [3.二阶卡尔曼滤波](#3二阶卡尔曼滤波)
      - [3.1状态向量和观测向量](#31状态向量和观测向量)
      - [3.2状态转移矩阵](#32状态转移矩阵)
      - [3.3过程噪声协方差矩阵和测量噪声协方差矩阵](#33过程噪声协方差矩阵和测量噪声协方差矩阵)




## 一、前言

最近在研究Franka Panda这款机械臂，在实际操作之前，我一直在思考如何更好的与机械臂交互。对于人类来讲，在抓取一个物体时，直接去抓就可以了，那么对于机械臂来讲，如何做到看到什么就直接抓取什么呢？

因此有了一个大胆的想法：做一个eye-to-hand，只不过这个eye是人眼，hand是机械臂的手。但是只有一步eye-to-hand还不够，缺少控制信号，因此在eye-to-hand的同时，需要将人手的姿态同步到机械臂上。

本文针对手势部分进行了总结与分析。



## 二、相机选型

如果只需要做“手势识别”与“手势分类”，笔记本电脑自带的摄像头就够了。

如果需要进行手势的立体视觉相关的操作，需要用到双目相机或者深度相机。由于实验室中已经购买了Intel Realsense L515(其他项目用的)，因此本文用的是深度相机，但是如果真的要做三维追踪，我推荐自己选帧率更高一点的双目相机，通过三角测量获得手部特征点的三维点，而且价格更便宜。



## 三、运行demo

项目文件地址：[EricSanchez的github手势识别库](https://github.com/EricSanchezok/hand_gesture_recognition.git)

项目文件结构：

![image-20230626143233242](images/image-20230626143233242.png)

1.main.py

当你第一次下载下来库文件后，只要你的电脑自带摄像头，可以直接运行main.py文件体验一下，该文件会默认启动videocapture(0)，并且在库中已经存在11种手势：

![image-20230626203921236](images/image-20230626203921236.png)




运行main.py后显示的图像如下，左上角是帧率，右上角是识别到的标签值：

![main](images/main.gif)

可以看到手势识别是立体的，不会因为旋转、平移或者是遮挡而影响识别。



当你把save_data_mode = False改为True时，可以开启保存数据模式：

![image-20230626184311735](images/image-20230626184311735.png)

你可以输入键盘上的'0~9'和'a'来调整编号'0~10'，对应了11种手势，具体可以查看文件'Data_Saver.py'，再次按下k键后会将每帧的数据保存在'dataset/'中。

保存模式会开启镜像模式，同时保存左右两幅图像的数据，保证网络可以识别左右手。



2.L515-main.py

另外还有一个demo文件：L515-main.py，是根据深度相机做了手部三维追踪，当然，只有在你连接了IntelRealsense L515才可以执行。

L515-main.py也可以开启保存模式，当你正常运行时，它不仅可以识别手势，当你的手势为“1”时，会自动追踪食指的位置，这是为了后续追踪食指的三维空间做准备。

![finger-track](images/finger-track.gif)

三维追踪的结果可以看下图：

![3D-points](images/3D-points.gif)

三维点的数据嵌套了两层滤波，第一层是滑动加权平均滤波，第二层则是卡尔曼滤波，目标的运动模式假设为匀加速运动，如果只看深度值的滤波算法结果如下图：

![depth-kalman](images/depth-kalman.gif)

实际上嵌套滑动加权平均滤波后的卡尔曼滤波已经有了十分明显的滞后性了，但是静态很稳定，可以抵消大部分由于测量造成的稳态误差，可以理解为消除人手自身的抖动。

其实想玩三维贪吃蛇的，但是matplot并没有透视，所以演示个Y-Z平面的贪吃蛇吧：

![game-play](images/game-play.gif)

## 四、mediapipe手势识别

谷歌的mediapipe有一套完整的人体识别的解决方案，本文用到了Hand Landmark

![image-20230626193153648](images/image-20230626193153648.png)

上图的左下角可以看到谷歌是做了Hand Gesture Recognition，对手势有了一些基础分类，但是实际上它并不能做到三维的手势分类，在它的网站上有demo可以直接调用你的设备的摄像头进行识别，测试之后你会发现：当你正面比耶的时候它还可以识别出这是"victory"的手势，侧过来就识别不到了：

<center class="half">
    <img src="images/image-20230626193440055.png" width="400"/>
    <img src="images/image-20230626193513964.png" width="400"/>
</center>




但是我们的分类器发挥很稳定：

<center class="half">
    <img src="images/image-20230626193818754.png" width="400"/><img src="images/image-20230626193853770.png" width="400"/>
</center>


关于mediapipe，网络上有非常多的案例和教程，这里就不再过多讲述了。



## 五、基于MLP和PointNet的手势分类

### 1.数据结构和类型

在做手势分类的网络前，一定要确定数据类型是什么样的。打开Hand_Detector.py文件，其中定义了mediaPipe_Hand_Detector类，其中的get_landmarks方法就返回了我们需要的数据：

```
class mediaPipe_Hand_Detector:
    def __init__(self, static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.1) -> None:
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode, max_num_hands, model_complexity, min_detection_confidence, min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.handLmsStyle = self.mpDraw.DrawingSpec(color=(0,0,255), thickness=5)
        self.handConStyle = self.mpDraw.DrawingSpec(color=(0,255,0), thickness=5)

    def get_landmarks(self, color_image, draw_fingers=True):

        results = self.hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks and draw_fingers:
            for handLms in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(color_image, handLms, self.mpHands.HAND_CONNECTIONS, self.handLmsStyle, self.handConStyle)

        if results.multi_hand_world_landmarks and results.multi_hand_landmarks:
            handLandmarks_points = []
            handworldLandmarks_points = [] 

            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    handLandmarks_points.append(lm.x)
                    handLandmarks_points.append(lm.y)
                    handLandmarks_points.append(lm.z)

            handLandmarks_points = np.array(handLandmarks_points).reshape(1, -1)

            for handLms in results.multi_hand_world_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    handworldLandmarks_points.append(lm.x)
                    handworldLandmarks_points.append(lm.y)
                    handworldLandmarks_points.append(lm.z)

            handworldLandmarks_points = np.array(handworldLandmarks_points).reshape(1, -1)

        else:
            handworldLandmarks_points = None
            handLandmarks_points = None

        return handLandmarks_points, handworldLandmarks_points
```

但是实际上我们用的是handworldLandmarks_points作为数据，另一个数据是为了在图片中画手部的识别结果用的。

也就是说手部的数据是手指21个特征点的x,y,z的坐标，也就是一个(1,63)的张量：

![img](images/2188126949-619d986c6b1d9.png)

当你打开dataset文件夹中的data_*.csv文件时也可以看到，每一行对应着一个手部姿态，而最后一列则是这个姿态对应的标签：

![image-20230626195054415](images/image-20230626195054415.png)

### 2.MLP手势分类

#### 2.1数据预处理——MLP

当使用MLP作为分类器时，数据处理原则依据Data_Preprocess.py中的landmarks_to_linear_data()方法：

```
def landmarks_to_linear_data(data_process):

    data = data_process.copy()

    if type(data) == pd.DataFrame:
        #将所有object转换为float
        data = data.astype(float)

        num_columns = data.shape[1]

        #给dataframe加上列名
        column_names = []
        for i in range(num_columns-1):
            column_names.append(f'{i//3}{"xyz"[i%3]}')
        column_names.append('label')

        data.columns = column_names

        # 进行独热编码
        one_hot_encoded = pd.get_dummies(data['label'], prefix='label')

        # 将编码后的列与原数据合并
        data = pd.concat([data.drop('label', axis=1), one_hot_encoded], axis=1)

        for i in range(21):
            data[f'{i}x'] = data[f'{i}x'] - data['0x']
            data[f'{i}y'] = data[f'{i}y'] - data['0y']
            data[f'{i}z'] = data[f'{i}z'] - data['0z']

        data = data.sample(frac=1) 

        X = data.iloc[:, :63].values  # 获取输入数据（特征）
        y = data.iloc[:, 63:].values  # 获取输出数据（标签）

        # 将数据转换为 PyTorch 张量
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        X = normalize(X)

        
    
    if type(data) == np.ndarray:

        for i in range(data.shape[1]):
            if i % 3 == 0:
                data[:, i] = data[:, i] - data[:, 0]
            elif i % 3 == 1:
                data[:, i] = data[:, i] - data[:, 1]
            elif i % 3 == 2:
                data[:, i] = data[:, i] - data[:, 2]


        X = torch.from_numpy(data)
        X = X.to(torch.float32)
        X = normalize(X)
        X = X.unsqueeze(0)

        y = None

    
    return X, y
```

分为两部分：

如果输入数据是DataFrame，意味着训练时的输入，主要的处理手段有：将标签进行独热编码、将第0个点作为原点所有点减去第0个点的坐标、标准化。

为什么要对标签进行独热编码

是因为最开始的时候，无论是用MLP还是其他网络做分类器，我在最后一层都没有添加softmax，这样的话，假如摆出的手势不属于任何标签，不会因为softmax把一个偏离1的预测值强行拉到1，这样可以通过一个阈值把这个不属于任何一个标签的手势排除掉。（但是在这个版本中我还是添加上了softmax）



如果输入的数据是numpy数组，意味着是在识别过程中的数据，数据处理的方式和上面类似。

#### 2.2网络结构——MLP

网络结构在MLP_Model.py文件中：

```
class MLP(nn.Module):
    def __init__(self, in_features, dropout):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 11)
        )


    def forward(self, x):
        return F.softmax(self.net(x), dim=1)
```

#### 2.3训练网络——MLP

训练mlp的过程在train_mlp.ipynb中，当你在训练的时候只需要注意有没有cuda就可以了其他的都很常规。

```
# 创建 MLP 模型实例
model = MLP_Model.MLP(63, 0.1)

model.cuda(device=device)

# 定义损失函数和优化器
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)

num_epochs = 20
num_samples = X.shape[0]

batch_size = 32


for epoch in range(num_epochs):
    train_loss = 0.0
    for i in range(0, num_samples, batch_size):
        input = X[i:i+batch_size]
        label = y[i:i+batch_size]
 
        # 前向传播
        model.train()
        output = model(input)

        l = loss(output, label)
        
        # 反向传播和优化
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.item()

    train_loss /= num_samples

    with torch.no_grad():
        model.eval()
        y_pred = model(Xval)
        val_loss = loss(y_pred, yval).item()


    # 打印每个 epoch 的损失
    print(f'Epoch {epoch+1}/{num_epochs}, train_Loss: {train_loss}, val_Loss: {val_loss}')
```

### 3.PointNet手势分类

#### 1.PointNet的结构



#### 2.PointNet的数据预处理



#### 3.PointNet的训练过程



#### 4.PointNet和MLP的对比

实际上在前期，我只采集到了5000多份数据的时候，PointNet的性能要远远强于MLP，并且有很好的平移和旋转不变性，但是在后面我邀请几位同学来采集他们的手部数据，将数据量扩充到了60000多份之后，MLP的稳定性又远远强于了PointNet，具体的对比如下：





这种现象似乎很难解释，因为MLP本身并不具备卷积的旋转和平移不不变性，它只能去判断某个特征相较于其他特征是否更重要，但是似乎在数据量增大了之后，在三维的手势中，包含了遮挡、旋转、缩放等变换后，MLP居然稳定性更好。

我认为唯一的解释就是我的数据做的太好了，把MLP的优势放大，此时PointNet的劣势就比较明显了，由于只有21个三维点，其中任何一个点的波动，PointNet都能敏锐的感受到，造成结果的不稳定性，并且标签数也太少，目前只有11个手势，PointNet对不同手势之间的过度部分做的并没有MLP圆滑，因此我认为这是PointNet在数据量增大的情况下败下阵来的原因，PointNet还是更适合做接近满足大数定理的特征的数据。





## 六、手势三维追踪

### 1.窗口深度搜索



### 2.滑动加权平均滤波



### 3.二阶卡尔曼滤波

我们假设人手的运动模式为匀加速直线运动(实际上我是从最简单的匀速直线运动开始的，但是效果并不好)，卡尔曼滤波的初始化参数如下：

#### 3.1状态向量和观测向量

目标包含三维位置和三维速度，因此状态向量是6维

而深度相机测量的值只有目标三维位置，因此观测向量维度为3维

```
kf = KalmanFilter(dim_x=6, dim_z=3)  # 状态向量维度为6，观测向量维度为3
```

#### 3.2状态转移矩阵

由于我们假设目标是匀加速运动，因此状态转移矩阵为：

```
#假设目标的运动模式为匀速运动，因此状态转移矩阵为：
kf.F = np.array([[1, 0, 0, dt, 0, 0.5*dt**2],
                [0, 1, 0, 0, dt, 0.5*dt**2],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])
```

其中dt是每次循环的间隔时间

#### 3.3过程噪声协方差矩阵和测量噪声协方差矩阵

这个值就是试出来的，没有什么标准答案

```
q = 0.01  # 过程噪声方差
# 定义过程噪声协方差矩阵
kf.Q = np.eye(6) * q  # q为过程噪声方差

r = 0.1  # 测量噪声方差
# 定义测量噪声协方差矩阵
kf.R = np.eye(3) * r  # r为测量噪声方差
```

