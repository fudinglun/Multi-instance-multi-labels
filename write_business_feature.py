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


# buz_memory = pickle.load(open("buz.p", "rb"))
#	pickle.dump(x, open("buz_cache/{}.p".format(key), "wb"))

target_value = target_value_extract("../yelp_data/train.csv")
input_value = input_value_extraction(INPUT_PATH)
keys = list(target_value.keys())
buz_memory = {}
pretrained_model = get_pretrained_model("resnet152_b")
transforms = get_transform()

for i in range(2000):
	key = keys[i]
	ids = input_value[key]
	x = get_business_feature(ids, pretrained_model, transforms)/len(ids)
	# buz_memory[key] = x
	pickle.dump(x, open("buz_cache/{}.p".format(key), "wb"))
	print(i)
# pickle.dump(buz_memory, open("business_feature_2048_7_7.p", "wb"))
