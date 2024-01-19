## inp
`inp` 是一個包含隨機數的 PyTorch 變數（Variable），它代表了線性層的輸入。這個輸入有一個大小為 (1, 10) 的張量，表示一個批次中有一個樣本，每個樣本具有 10 個特徵。
  
    inp = Variable(torch.randn(1, 10))

## grad_fn=<AddmmBackward0>
`grad_fn=<AddmmBackward0>`表示這個張量是由一個矩陣-矩陣乘法操作（add matrix multiplication backward）生成的，這是 PyTorch 中用於計算梯度的一部分。
