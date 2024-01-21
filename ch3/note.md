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
