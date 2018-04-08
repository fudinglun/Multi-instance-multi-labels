import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import csv

# print(torch.cuda.is_available())

model_conv = torchvision.models.resnet152(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

model = nn.Sequential(model_conv.conv1,
	model_conv.bn1,
	model_conv.relu,
	model_conv.maxpool,
	model_conv.layer1,
	model_conv.layer2,
	model_conv.layer3,
	model_conv.layer4,
	model_conv.avgpool)


data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_image_feature(photo_id):
	path = "../yelp_data/train_photos/{}.jpg".format(photo_id)
	photo = Image.open(path)
	photo = Variable(data_transforms(photo).unsqueeze(0), requires_grad=False)
	if torch.cuda.is_available():
		model.cuda()
		photo.cuda()
	x = model(photo)
	x = x.view(x.size(0), -1)
	data = x.data.numpy()[0]
	return data
