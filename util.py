import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import pdb
import csv
import datetime


"""
Given a path, usual: INPUT_PATH = "../yelp_data/train_photo_to_biz_ids.csv"
return a dictionary map a business id to a list of photo ids
"""
def input_value_extraction(path):
	dic = {}
	with open(path) as f:
		next(f)
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			photo = row[0]
			buz = row[1]
			if buz in dic:
				dic[buz].append(photo)
			else:
				dic[buz] = [photo]
	return dic


"""
Given a path, usually "../yelp_data/train.csv"
Return a dictionary map a business id to a target vector of length 9
"""
def target_value_extract(path):
	dic = {}
	with open(path) as f:
		next(f)
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			vals = row[1].split(' ')
			y = np.zeros(9)
			for i in vals:
				if i == '':
					continue
				y[int(i)] = 1
			dic[row[0]] = y
	return dic


"""
Get image level features from the pre_trained model
"""
def get_image_features(ids, model, data_transforms):
	images = []
	for i in ids:
		path = "../yelp_data/train_photos/{}.jpg".format(i)
		photo = data_transforms(Image.open(path))
		images.append(photo)
	images = torch.stack(images)
	if torch.cuda.is_available():
		images = images.cuda()
	images = Variable(images, requires_grad=False)
	x = model(images)
	x = x.view(x.size(0), -1)
	return x.data



"""
Get selected pretrained model
"""
def get_pretrained_model(name="resenet152"):
	if name == "resenet152":
		return get_pretrained_resnet152()




"""
Get pretrained resnet152
"""
def get_pretrained_resnet152():
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

	if torch.cuda.is_available():
		model = model.cuda()
		print("Set pretrained model cuda")
	return model