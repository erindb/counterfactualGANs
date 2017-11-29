import numpy as np
import pandas as pd

from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable

attr_labels = pd.read_table('./counterfactualGANs/list_attr_celeba.txt', 
	delim_whitespace=True, usecols=['File_Name', 'Smiling'])

print(attr_labels.iloc[0])

preprocess = transforms.Compose([
	transforms.ToTensor()
])

img_pil = Image.open('img1.jpg')
img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)
print(img_tensor)