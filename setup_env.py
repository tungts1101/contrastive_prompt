import os
import urllib.request

def download_file(url, folder):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, os.path.basename(url))
    if os.path.exists(filename): return
    urllib.request.urlretrieve(url, filename)

# Download pretrained ViT-B/16 with MAE
download_file("https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth", "checkpoints")
