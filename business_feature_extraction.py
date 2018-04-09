import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import csv
from image_feature_extraction import get_image_feature
import pdb
from sklearn.model_selection import train_test_split
import datetime




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

target_value = target_value_extract("../yelp_data/train.csv")

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

input_value = input_value_extraction("../yelp_data/train_photo_to_biz_ids.csv")

keys = list(target_value.keys())
train_key, test_key = train_test_split(keys)




class BaseLine(nn.Module):
	def __init__(self):
		super(BaseLine, self).__init__()
		self.fc1 = nn.Linear(8*2048, 9)

	def forward(self, x):
		return F.sigmoid(self.fc1(x))


model = BaseLine()
# model.load_state_dict(torch.load("model2018-04-09.pt"))
if torch.cuda.is_available():
	print("GPU is available")
	model = model.cuda()

optimizer = optim.Adam(model.parameters())
criteria = nn.BCELoss()


def train_baseline(epoch):
	model.train()
	for e in range(epoch):
		j = 0
		running_loss = 0.0
		for key in train_key:
			for num in range(len(input_value[key]) // 8 + 1):
				x = np.array([])
				y = target_value[key]
				ids = np.random.choice(input_value[key], 8)
				for i in ids:
					data = get_image_feature(i)
					x = np.concatenate((x, data))
				x = torch.from_numpy(x).float()
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
			print("epoch: {}, key: {}, with loss: {}".format(e, j, loss.data[0]))
			running_loss += loss.data[0]
			j += 1
		now = datetime.datetime.now()
		torch.save(model.state_dict(), "model{}.pt".format(str(now.date())))
		print("finish epoch {}".format(e))
		print("Loss of this epoch is {}".format(running_loss/2000))


train_baseline(5)












