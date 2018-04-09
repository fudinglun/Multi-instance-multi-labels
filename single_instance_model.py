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


class SingleInstanceBaseLine(nn.Module):
	def __init__(self):
		super(SingleInstanceBaseLine, self).__init__()
		self.fc1 = nn.Linear(2048, 9)
		self.sig = nn.Sigmoid()


	def forward(self, x):
		return self.sig(self.fc1(x))




target_value = target_value_extract("../yelp_data/train.csv")
model = SingleInstanceBaseLine()
# model.load_state_dict(torch.load("model2018-04-09.pt"))
if torch.cuda.is_available():
  print("GPU is available")
  model = model.cuda()

optimizer = optim.Adam(model.parameters())
criteria = nn.BCELoss()


epoch = 1
batch_size = 10
pretrained_model = get_pretrained_model()
transforms = get_transform()
for e in range(epoch):
	with open(INPUT_PATH) as f:
		next(f)
		reader = csv.reader(f, delimiter=',')
		count = 0
		running_loss = 0.0
		while True:
			try: 
				ids = []
				targets = []
				for b in range(batch_size):
					photo_id, buz_id = next(reader)
					ids.append(photo_id)
					targets.append(target_value[buz_id])
				x = get_image_features(ids, pretrained_model, transforms)
				y = torch.from_numpy(np.vstack(targets)).float()
				if torch.cuda.is_available():
					x = x.cuda()
					y = y.cuda()
				x, y = Variable(x), Variable(y)
				optimizer.zero_grad()
				output = model(x)
				loss = criteria(output, y)
				loss.backward()
				optimizer.step()
				running_loss += loss.data[0]
				count += 1
				if count % 200 == 0:
					print("Epoch: {}, iteration: {}, average loss: {}".format(e, count, running_loss / 200))
					running_loss = 0.0
				if count % 2000 == 0:
					now = datetime.datetime.now()
					torch.save(model.state_dict(), "single_instance_model{}.pt".format(str(now.date())))
			except csv.Error:
				print("Error")
			except StopIteration:
				print("Iteration End")
				break





