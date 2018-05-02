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
from util import input_value_extraction, target_value_extract, get_pretrained_model, get_image_features

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
		x = get_image_features(ids[:100], model, data_transforms)
		x = torch.sum(x, 0)
		more = get_business_feature(ids[100:], model, data_transforms)
		x = torch.add(x, 1, more)
		return x
	else:
		x = get_image_features(ids, model, data_transforms)
		x = torch.sum(x, 0)
		return x

class AverageInstanceNonlinear(nn.Module):
	def __init__(self, input_d, hidden_d):
		super(AverageInstanceNonlinear, self).__init__()
		self.fc1 = nn.Linear(input_d, hidden_d)
		self.fc2 = nn.Linear(hidden_d, 9)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		return self.sigmoid(self.fc2(self.sigmoid(self.fc1(x))))


target_value = target_value_extract("../yelp_data/train.csv")
input_value = input_value_extraction(INPUT_PATH)
keys = list(target_value.keys())
model = AverageInstanceNonlinear(100352, 2048)
model.load_state_dict(torch.load("research_model2018-04-30.pt"))
if torch.cuda.is_available():
  print("GPU is available")
  model = model.cuda()

###best lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=0.005)
criteria = nn.BCELoss()


buz_memory = {}

print("LR Research Model running")
epoch = 1000
pretrained_model = get_pretrained_model("resnet152_b")
transforms = get_transform()



model.train()
for e in range(epoch):
	for batch in range(60):
		key_pool = keys[batch*25:(batch+1)*25]
		x_batch = []
		y_batch = []
		for key in key_pool:
			if key in buz_memory:
				x = buz_memory[key]
			else:
				ids = input_value[key]
				x = get_business_feature(ids, pretrained_model, transforms) / len(ids)
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
		
	now = datetime.datetime.now()
	torch.save(model.state_dict(), "LR_research_model{}.pt".format(str(now.date())))



####TEST
model.eval()
thresholds = [0.4,0.5,0.6]
total = 500
result = [[0,0],[0,0],[0,0]]
for i in range(1500, 2000):
	key = keys[i]
	ids = input_value[key]
	x = get_business_feature(ids, pretrained_model, transforms) / len(ids)
	y = target_value[key]
	if y.sum() == 0:
		total -= 1
		continue
	if torch.cuda.is_available():
		x = x.cuda()
	x = Variable(x)
	output = model(x)
	for t in range(len(thresholds)):
		recall_total = 0
		recall_count = 0
		precision_total = 0
		precision_count = 0	
		for i in range(9):
			if y[i] == 1:
				recall_total += 1
				if output.data[i] >= thresholds[t]:
					recall_count += 1
			if output.data[i] >= thresholds[t]:
				precision_total += 1
				if y[i] == 1:
					precision_count += 1
		result[t][0] += recall_count / recall_total
		result[t][1] += precision_count / precision_total
for t in range(len(thresholds)):
	print("Threhods: {}, recall: {}, precision: {}".format(thresholds[t], result[t][0]/total, result[t][1]/total))




