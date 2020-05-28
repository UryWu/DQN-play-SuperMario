# DQN-play-SuperMario
Use the deep reinforcement learning algorithm DQN to play the game SuperMario.  
The game environment is come from https://github.com/justinmeister/Mario-Level-1.git  
The agent performance is not good enough.Actually there is a gap between my expectations and the fact.This kind of learning algorithms are magical and hard to control.There must be some internal logic between the agent performance and a bunch of hyperparameters.And it is important to note that the supermariobros game is complicated than Atari games in gym,the game frame is more rich and has more changes. 

## This project is to be continue!
Anyway,you can just clone this repository and run "play_mario.py"
## My environment(UryWu)
* python 3.6

* opencv-python==4.2.0

* Pillow==7.1.2

* scipy==1.4.1

* pytorch==1.5.0

  you should install torch by commend: 

  ```shell
  pip install pytorch -f https://download.pytorch.org/whl/torch_stable.html
  ```

* [the supermario game environment](https://github.com/justinmeister/Mario-Level-1.git)



## My Debug logs(UryWu)

### Bug1:

**description：**

```bash
AttributeError: module 'scipy.misc' has no attribute 'imresize' 
```

**location：**

```bash
File "F:\Projects\DQN-play-SuperMario\data\env.py", line 94, in reset
    state = scipy.misc.imresize(state, (self.resize_x, self.resize_y))
```

**solution：**
更换env.py里的这样的相关行，因为在1.3.0版本的scipy之后，去除了scipy.misc.imresize模块，只能使用其它的库更换图片大小

```python
state = scipy.misc.imresize(state, (self.resize_x, self.resize_y))
```

改为为：

```python
state = state.resize((self.resize_x, self.resize_y), Image.ANTIALIAS)
```

将：

[reference](https://stackoverflow.com/questions/56204985/how-to-fix-scipy-misc-has-no-attribute-imresize/56205147)

### Bug2：

**description：**

```
TypeError: 'tuple' object cannot be interpreted as an integer
```

**location：**

```bash
File "F:\Projects\DQN-play-SuperMario\data\env.py", line 95, in reset
    state = state.resize((self.resize_x, self.resize_y), Image.ANTIALIAS)
```

**solution：**
bug1那里是使用from PIL import Image我们需要把state转化成Image格式，如下：

```python
state = Image.fromarray(np.uint8(state))
```


然后再resize。



### Bug3：

**description：**

```bash
TypeError: 'Image' object is not subscriptable
```

**location：**

```bash
File "F:\Projects\DQN-play-SuperMario\data\env.py", line 72, in rgb2gray
    np.dot(image[..., :3], [0.299, 0.587, 0.114])
```

**solution：**因为传入的图片image是<class 'PIL.Image.Image'>类型，无法切片，所以需要转换成numpy的ndarray类型，代码如下：

```python
image = np.asarray(image)
```

然后再使用np.dot点乘



### Bug4：

**description：**

```bash
RuntimeError: Expected 4-dimensional input for 4-dimensional weight [16, 4, 8, 8], but got 5-dimensional input of size [1, 4, 84, 84, 3] instead
```

**location：**

```bash
File "F:\Projects\DQN-play-SuperMario\net_pytorch.py", line 24, in forward
    output=F.relu(self.conv1(input))
```

**solution:**
net_pytorch.py, line 44

```python
input=Variable(input.unqueeze(0))
```

改为：

```python
input=Variable(input)
```

把input的维度[1, 4, 84, 84, 3]改为[4, 84, 84, 3]，除去第一个空维度。



### Bug5：

**description：**

```bash
RuntimeError: Given groups=1, weight of size [16, 3, 8, 8], expected input[4, 84, 84, 3] to have 3 channels, but got 84 channels instead
```

**location：**

```bash
File "F:\Projects\DQN-play-SuperMario\net_pytorch.py", line 26, in forward
    output=F.relu(self.conv1(input))
```

**solution：**在上一个错误我只是取消了第一个空维度，现在第一层卷积是3个通道，（我改了第一层卷积为3个通道），而输入的input
是[4个batchsize, 84宽, 84高, 3通道]，需要把通道维换到batchsize维后面，在net_pytorch.py的46行代码input=Variable(input)
的前面加上：input = input.permute(0, 3, 1, 2)，这样input的shape变为[4个batchsize, 3通道, 84宽，84高]

### Study records

[1、how to transpose dim in torch tensor](https://blog.csdn.net/weixin_44613063/article/details/89521464)

