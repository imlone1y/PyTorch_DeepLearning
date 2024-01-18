# anaconda PyTorch 環境架設(ubuntu 20.04)

## 1. 下載 anaconda

> [anaconda download](https://www.anaconda.com/download#windows)

下載完 anaconda 後，進入 Anaconda Prompt

## 2. 使用 Anaconda Prompt(anaconda3) 創建 python3.7 環境

創造一個 conda for PyTorch 環境

    conda create -n pytorch python=3.7

切進環境

    conda activate pytorch

## 3.安裝 PyTorch

進入 Pytorch 官網查詢自己電腦型號的 PyTorch 下載碼

> [Start Locally | PyTorch](https://pytorch.org/get-started/locally/)

![image](https://github.com/imlone1y/PyTorch_DeepLearning/assets/136362929/56cf5b25-19ae-4bb5-b9e2-ea51c2296fa7)

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

安裝完後驗證

    conda activate pytorch
    
    python
    
    import torch
    
未顯示錯誤訊息即為安裝成功

    import torchvision
    
    torchvision.__version__

> 應顯示版本名稱

驗證 CUDA 是否可用

    torch.cuda.is_available()

> True

    torch.cuda.get_device_name()

> 應顯示顯卡名稱

    torch.cuda.device_count()

> 1
CUDA ID

    torch.cuda.current_device()

CUDA 裝置名稱

    torch.cuda.get_device_name(0)


