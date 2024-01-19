## inp
`inp` 是一個包含隨機數的 PyTorch 變數（Variable），它代表了線性層的輸入。這個輸入有一個大小為 (1, 10) 的張量，表示一個批次中有一個樣本，每個樣本具有 10 個特徵。
  
    inp = Variable(torch.randn(1, 10))
