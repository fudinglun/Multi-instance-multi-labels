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

INPUT_PATH = "../yelp_data/train_photo_to_biz_ids.csv"

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

def get_pretrained_model():
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

def get_transform():
	data_transforms = transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	return data_transforms

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

def get_business_feature(ids, model, data_transforms):
	if len(ids) > 100:
		x = get_image_features(ids[:100], model, data_transforms)
		x = torch.sum(x, 0)
		more = get_business_feature(ids[100:], model, data_transforms)
		x = torch.add(x, 1, more)
		return x
	else:
		x = get_image_features(ids, model, data_transforms)
		x = torch.sum(x, 0)
		return x


class AverageInstanceBaseLine(nn.Module):
	def __init__(self):
		super(AverageInstanceBaseLine, self).__init__()
		self.fc1 = nn.Linear(2048, 9)
		self.sig = nn.Sigmoid()


	def forward(self, x):
		return self.sig(self.fc1(x))



target_value = target_value_extract("../yelp_data/train.csv")
input_value = input_value_extraction(INPUT_PATH)
keys = list(target_value.keys())
model = AverageInstanceBaseLine()
if torch.cuda.is_available():
  print("GPU is available")
  model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.01)
criteria = nn.BCELoss()

epoch = 10
pretrained_model = get_pretrained_model()
transforms = get_transform()
for e in range(epoch):
	count = 0
	for key in keys:
		ids = input_value[key]
		x = get_business_feature(ids, pretrained_model, transforms)
		pdb.set_trace()
		x = x/len(ids)
		y = target_value[key]
		y = torch.from_numpy(y).float()
		if torch.cuda.is_available():
			x = x.cuda()
			y = y.cuda()
		x, y = Variable(x), Variable(y)
		optimizer.zero_grad()
		output = model(x)
		loss = criteria(output, y)
		loss.backward()
		optimizer.step()
		print("Epoch: {}, iteration: {}, loss: {}".format(e, count, loss.data[0]))
		count += 1
	now = datetime.datetime.now()
	torch.save(model.state_dict(), "average_instance_model{}.pt".format(str(now.date())))








