import numpy as np
import pandas as pd

#from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable

class AlexNet(nn.Module):

    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # the original model ended up with 6*6*256, and so do we
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 5 * 5, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 5 * 5)
        x = self.classifier(x)
        return x

attr_labels = pd.read_table('/Users/alexgoodman/desktop/list_attr_celeba.txt',
	delim_whitespace=True, usecols=['File_Name', 'Smiling'])

preprocess = transforms.Compose([
	transforms.ToTensor()
])

## This stuff will be necessary for training but not for actual use later
#img_pil = Image.open('no_smile.png')
#img_tensor = preprocess(img_pil)
#img_tensor.unsqueeze_(0)
#print(img_tensor)

img_tensor = torch.from_numpy(np.random.rand(64, 3, 64, 64)).float()
#img_tensor = #load in from generator/dataset
img_tensor = Variable(img_tensor, requires_grad=True)

# x will be input batch tensor
#x = something
# 1 = smiling, 0 = not smiling
y = attr_labels['Smiling'].apply(lambda x: int(x + 1 != 0))

model = AlexNet()
loss_fn = nn.CrossEntropyLoss()
learn_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

y_pred = model.forward(img_tensor)
print(y_pred)

# Training step
for i in range(1000):
    # Forward pass
    y_pred = model.forward(x)

    # Compute and print loss
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # Zero gradients before backward pass
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

# Save the model after training
torch.save(model.state_dict(), './smile_model')
