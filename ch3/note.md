```python
myLayer = Linear(in_features=10,out_features=5,bias=True) # 輸入特徵數為 10，輸出特徵數為 5，並啟用偏差（bias）
inp = Variable(torch.randn(1,10))
myLayer(inp) # 執行線性轉換
```
output:

    tensor([[ 0.3442,  1.3424, -0.0624, -0.2083, -0.6045]],grad_fn=<AddmmBackward0>)

## inp
`inp` 是一個包含隨機數的 PyTorch 變數（Variable），它代表了線性層的輸入。這個輸入有一個大小為 (1, 10) 的張量，表示一個批次中有一個樣本，每個樣本具有 10 個特徵。
  
    inp = Variable(torch.randn(1, 10))

## `grad_fn=\<AddmmBackward0\>`
`grad_fn=<AddmmBackward0>`表示這個張量是由一個矩陣-矩陣乘法操作（add matrix multiplication backward）生成的，這是 PyTorch 中用於計算梯度的一部分。

***

```python
class MyFirstNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MyFirstNetwork,self).__init__() 
        self.layer1 = nn.Linear(input_size,hidden_size) # 透過這兩個線性層，模型可以學習從輸入到輸出的映射。在每一層之間，使用 ReLU 激活函數（torch.relu）來引入非線性。
        self.layer2 = nn.Linear(hidden_size,output_size)
    def __forward__(self,input): 
        out = self.layer1(input) # 執行了線性轉換操作，將輸入資料映射到隱藏層（hidden layer）的空間。
        out = nn.ReLU(out)
        out = self.layer2(out) 
        return out
```

## `super(MyFirstNetwork,self).__init__()`

這是 Python 中用於呼叫父類別方法的標準方式。

`super()` 返回一個臨時對象，該對象用於調用父類別的方法。在這個情況下，它確保 `MyFirstNetwork` 的父類別 `nn.Module` 的初始化方法被正確地呼叫。

簡單來說，這一行程式碼確保您的自定義神經網路在初始化時會執行父類別 `nn.Module` 的初始化方法。

***

```python
loss = nn.MSELoss() # 創建一個均方誤差的損失函數，並將其存儲在 loss 變數中。
input = Variable(torch.randn(3, 5), requires_grad=True)
target = Variable(torch.randn(3, 5))
output = loss(input, target) # 計算模型輸出 (input) 與目標值 (target) 之間的均方誤差損失。
output.backward() # 執行反向傳播，計算損失對於輸入變數 input 的梯度。這將使得 input 中的 requires_grad 為 True 的元素具有相對於損失的梯度值。
```
## `input = Variable(torch.randn(3, 5), requires_grad=True)`
這一行程式碼創建了一個大小為 (3, 5) 的張量（tensor），其中的數據是從標準正態分佈中抽樣而得。這個張量被包裝成一個 `Variable`，而 `requires_grad=True` 表示你希望計算這個變數的梯度。

## `target = Variable(torch.randn(3, 5))`
這一行程式碼創建了另一個大小相同的張量，作為目標值（ground truth）。同樣，這個張量也被包裝成一個 `Variable`，但由於沒有指定 `requires_grad`，預設是 `False`，因此不需要計算它的梯度。

## 反向傳播

反向傳播是深度學習中用於計算梯度的一個關鍵步驟，通常與損失函數和模型的參數有關。在 PyTorch 中，這是透過 `backward` 方法實現的。

以下是反向傳播的詳細解釋：

+ 損失函數計算： 首先，你需要有一個損失函數，用來衡量模型預測輸出與實際目標之間的差異。在你的例子中，使用的是均方誤差（MSE）損失函數。

+ 前向傳播： 將輸入資料通過模型（神經網路）進行前向傳播。這將得到模型的預測輸出。

+ 損失計算： 使用損失函數計算模型預測輸出與實際目標之間的差異，得到一個損失值。

+ 反向傳播： 調用 `backward` 方法，這將計算損失對於模型參數的梯度。在這個過程中，PyTorch會自動計算梯度並將其存儲在相應的張量的 `grad` 屬性中。

在你的程式碼中，`output.backward()` 正是這個步驟，它計算了損失對於 `input` 張量中 `requires_grad=True` 的元素的梯度。

梯度更新： 梯度計算完畢後，你可以使用優化器（optimizer）來更新模型的參數。優化器通常使用梯度下降等優化算法，將損失最小化，從而改善模型的預測。
總的來說，反向傳播是深度學習中的一個關鍵步驟，通過計算損失對於模型參數的梯度，使得模型能夠自動地學習適應性並不斷優化。這個過程是基於梯度下降的思想，通過最小化損失來提升模型的性能。




