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
from util import input_value_extraction, target_value_extract, get_pretrained_model, get_raw_image_features
import pickle

INPUT_PATH = "../yelp_data/train_photo_to_biz_ids.csv"


def get_transform():
	data_transforms = transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	return data_transforms

def get_business_feature(ids, model, data_transforms):
	if len(ids) > 100:
		x = get_raw_image_features(ids[:100], model, data_transforms)
		x = torch.sum(x, 0)
		more = get_business_feature(ids[100:], model, data_transforms)
		x = torch.add(x, 1, more)
		return x
	else:
		x = get_raw_image_features(ids, model, data_transforms)
		x = torch.sum(x, 0)
		return x

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(nn.Conv2d(2048, 512, 3),nn.ReLU(),nn.MaxPool2d(2))
		self.fc = nn.Linear(512*2*2, 9)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		result = self.layer1(x)
		result = result.view(x.size(0), -1)
		result = self.sigmoid(self.fc(result))
		return result

class MultiKernelCNN(nn.Module):
	def __init__(self):
		super(MultiKernelCNN, self).__init__()
		self.layer1 = nn.Sequential(nn.Conv2d(2048, 256, 3),nn.ReLU(),nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(nn.Conv2d(2048, 64, 4),nn.ReLU())
		self.fc = nn.Linear(512*2*2, 9)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		result1 = self.layer1(x)
		result2 = self.layer2(x)
		result1 = result1.view(x.size(0), -1)
		result2 = result2.view(x.size(0), -1)
		result = torch.cat((result1, result2), dim=1)
		result = self.sigmoid(self.fc(result))
		return result


target_value = target_value_extract("../yelp_data/train.csv")
input_value = input_value_extraction(INPUT_PATH)
keys = list(target_value.keys())
model = MultiKernelCNN()
if torch.cuda.is_available():
  print("GPU is available")
  model = model.cuda()

###best lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)
criteria = nn.BCELoss()


buz_memory = {}

print("Multi Kernel CNN Research Model running")
epoch = 200
pretrained_model = get_pretrained_model("resnet152_b")
transforms = get_transform()

def evaluation(model):
	print("Start Testing")
	model.eval()
	thresholds = [0.4,0.5,0.6]
	total = 500
	result = [[0,0],[0,0],[0,0]]
	for i in range(1500, 2000):
		key = keys[i]
		if key in buz_memory:
			x = buz_memory[key]
		else:
			ids = input_value[key]
			# x = get_business_feature(ids, pretrained_model, transforms)/len(ids)
			x = pickle.load(open("buz_cache/{}.p".format(key), "rb"))
			buz_memory[key] = x
		y = target_value[key]
		if y.sum() == 0:
			total -= 1
			continue
		x = [x]
		x = torch.stack(x)
		if torch.cuda.is_available():
			x = x.cuda()
		x = Variable(x)
		output = model(x)
		output = output.cpu()
		for t in range(len(thresholds)):
			recall_total = 0
			recall_count = 0
			precision_total = 0
			precision_count = 0	
			for i in range(9):
				if y[i] == 1:
					recall_total += 1
					if output.data[0][i] >= thresholds[t]:
						recall_count += 1
				if output.data[0][i] >= thresholds[t]:
					precision_total += 1
					if y[i] == 1:
						precision_count += 1
			result[t][0] += recall_count / recall_total if recall_total > 0 else 0
			result[t][1] += precision_count / precision_total if precision_total > 0 else 0
	for t in range(len(thresholds)):
		print("Validation: Threhods: {}, recall: {}, precision: {}".format(thresholds[t], result[t][0]/total, result[t][1]/total))

for e in range(epoch):
	model.train()
	for batch in range(60):
		key_pool = keys[batch*3:(batch+1)*3]
		x_batch = []
		y_batch = []
		for key in key_pool:
			if key in buz_memory:
				x = buz_memory[key]
			else:
				ids = input_value[key]
				x = pickle.load(open("buz_cache/{}.p".format(key), "rb"))
				# x = get_business_feature(ids, pretrained_model, transforms)/len(ids)
				buz_memory[key] = x
			y = target_value[key]
			y = torch.from_numpy(y).float()
			x_batch.append(x)
			y_batch.append(y)
		x_batch = torch.stack(x_batch)
		y_batch = torch.stack(y_batch)
		if torch.cuda.is_available():
			x_batch = x_batch.cuda()
			y_batch = y_batch.cuda()
		x_batch, y_batch = Variable(x_batch), Variable(y_batch)
		optimizer.zero_grad()
		output = model(x_batch)
		loss = criteria(output, y_batch)
		loss.backward()
		optimizer.step()
		print("Epoch: {}, Iteration: {}, Loss: {}".format(e, batch, loss.data[0]))
	# 	break
	# continue
	if e % 20 == 0:
		now = datetime.datetime.now()
		torch.save(model.state_dict(), "MultiKernel_CNN_model_epoch{}_{}.pt".format(e, str(now.date())))
		evaluation(model)





