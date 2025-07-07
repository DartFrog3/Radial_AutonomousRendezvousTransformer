# if not already installed
!git clone https://github.com/Stanford-CAESAR/art-aeroconf24.git
%cd art-aeroconf24
!pip install -r requirements.txt
%cd ..
!git clone https://github.com/mit-han-lab/radial-attention.git
# ignore restart error
%cd radial-attention
# versioning
!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
!pip install -r requirements.txt
!pip install flash-attn --no-build-isolation

# flashinfer
!pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/
from radial_attn.attn_mask import RadialAttention, MaskMap
%cd ..
%cd art-aeroconf24
# if necessary
# !pip install -U accelerate
# !pip install -U transformers
# !pip install -q peft accelerate einops
# imports
import torch, json, pathlib, math, random
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import DecisionTransformerConfig, GPT2Config, GPT2LMHeadModel
from transformer.art import AutonomousRendezvousTransformer # get actual model
import os, sys
import shutil
from torch.optim import AdamW
from accelerate import Accelerator
get dataset+weights
# local dataset from shared drive
# mount drive and access files from shortcut
from google.colab import drive
drive.mount('/content/drive')
ckpt_path = "/content/drive/MyDrive/ART-AeroConf24/checkpoint_rtn_art"
!cp -r {ckpt_path} "transformer/saved_files/checkpoints/"
# setup util
# copy only files to avoid additions from recursive cp call
def copy_files(src_dir, dst_dir):
    src = Path(src_dir)
    dst = Path(dst_dir)
    if not os.path.exists(dst):
        os.makedirs(dst)

    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
data_path = '/content/drive/MyDrive/ART-AeroConf24'
copy_files(data_path, 'dataset')