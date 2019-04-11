import numpy as np
from os import walk
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle
from random import randint
import seaborn as sns
from functools import reduce

sns.set()

class Settings():
	'''Class to hold/easily modify global settings'''

	def __init__(self,
				train_fp = './data/stage1_train/',
				test_fp = './data/stage1_test/',
				# Consistent (Width, height, channel)
				enforced_dims = (256, 256),
				train_validation_ratio = .7,
				dim_thres_vertical = .77,
				dim_thres_horizontal = 1.3,
				train_features_pkl = './data/saves/train_f.pkl',
				train_targets_pkl = './data/saves/train_t.pkl',
				test_features_pkl = './data/saves/test_f.pkl'):

		self.train_fp = train_fp
		self.test_fp = test_fp
		self.enforced_dims = enforced_dims
		self.train_validation_ratio = train_validation_ratio
		self.dim_thres_vertical = dim_thres_vertical
		self.dim_thres_horizontal = dim_thres_horizontal
		self.train_features_pkl = train_features_pkl
		self.train_targets_pkl = train_targets_pkl
		self.test_features_pkl = test_features_pkl

class NucleiImages():
	def __init__(self):
		self.settings = Settings()

	def get_data_fp(self):
		'''Return the fp for train data, train mask fp'''

		# First, walk dir to find all unique image folders
		train_ids = list(walk(self.settings.train_fp))[0][1]
		test_ids = list(walk(self.settings.test_fp))[0][1]

		# Helper functions to turn names into image filepaths
		train_image_fp = lambda name: self.settings.train_fp + name + f'/images/{name}.png'
		test_image_fp = lambda name: self.settings.test_fp + name + f'/images/{name}.png'
		
		# {'id': {'image': image_fp, 
		#		  'mask': ['mask_fp1, mask_fp2,...']},
		#  ...}
		train_dict = {name: {'image': train_image_fp(name), 'mask': None} for name in train_ids}
		test_dict = {name: {'image': test_image_fp(name)} for name in test_ids}
		
		# Find/add mask locations 
		for name in train_dict:
			train_mask_fp = self.settings.train_fp + name + '/masks/'
			mask_fps = list(walk(train_mask_fp))[0][2]
			mask_fps = list(map(lambda mask: train_mask_fp + mask, mask_fps))
			train_dict[name]['mask'] = mask_fps

		return train_dict, test_dict

	def image_statistics(self):
		'''This module is to inspect the distribution of image
		width/height data'''

		train_dict, _ = self.get_data_fp()

		widths = np.zeros(len(train_dict), dtype=int)
		heights = np.zeros(len(train_dict), dtype=int)

		for i, name in enumerate(train_dict):
			img_fp = train_dict[name]['image']
			im = imread(img_fp)
			w, h, _ = im.shape
			widths[i] = w
			heights[i] = h

		# Draw distribution of width, height, and their ratios
		fig, ax = plt.subplots(figsize = (15,10), nrows=3)
		sns.distplot(widths, ax=ax[0], axlabel='test')
		sns.distplot(heights, ax=ax[1])
		sns.distplot(widths/heights, ax=ax[2])

		# Set plot labels
		plt.tight_layout()
		ax[0].set(xlabel='width pixels', ylabel='freq')
		ax[1].set(xlabel='height pixels', ylabel='freq')
		ax[2].set(xlabel='w/h ratio', ylabel='freq')

		return fig, ax

	def process_save_data(self):
		# We will either keep images same (if dimension ratio = 1),
		# or randomly sample (if .91 > dimension ratio > 1.1 ), or
		# vertically sample 2 separate images if dim_ratio <= .91
		# else horizontally sample 2 separate images if dim_Ratio >= 1.1

		def rgb_to_gray(im):
			# https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert
			# Use Y' = 0.2989 R + 0.5870 G + 0.1140 B 
			return np.dot(im, [.2989, .5870, .1140])

		def flatten_masks(masks):
			# Turn stacks of binary masks into a single binary mask
			return reduce(np.maximum, masks)

		def resizer(im):
			# Resizes image to enforced_dims
			return resize(im, self.settings.enforced_dims,
							mode='constant',
							preserve_range=True,
							anti_aliasing=True)

		def random_sample(im, w, h, masks=None):
			# Randomly sample square whose dim is min(w, h)
			dim = [w, h]
			min_idx = dim.index(min(dim))
			min_pixel = dim[min_idx]

			if min_idx:
				x1, y1 = randint(0, w - min_pixel - 1), 0
			else:
				x1, y1 = 0, randint(0, h - min_pixel - 1)

			im1 = im[x1:x1 + min_pixel, y1:y1 + min_pixel]
			im1 = resizer(im1)

			if masks is None:
				return im1
			else:
				masks = masks[x1:x1 + min_pixel, y1:y1 + min_pixel]
				masks = resizer(masks)
				return im1, masks

		def vertical_sample(im, w, h, masks=None):
			# Sample top square, and bottom square based on min(w, h)
			im1, im2 = im[:, :w], im[:, -w:]
			im1 = resizer(im1)
			im2 = resizer(im2)

			if masks is None:
				return im1, im2
			else:
				masks1, masks2 = masks[:,:w], masks[:,-w:]
				masks1 = resizer(masks1)
				masks2 = resizer(masks2)
				return im1, im2, masks1, masks2

		def horizontal_sample(im, w, h, masks=None):
			# Sample left square, and right square based on min(w, h)
			im1, im2 = im[:h, :], im[-h:, :]
			im1 = resizer(im1)
			im2 = resizer(im2)

			if masks is None:
				return im1, im2
			else:
				masks = masks[:h,:], masks[-h:,:]
				masks1 = resizer(masks1)
				masks2 = resizer(masks2)
				return im1, im2, masks1, masks2

		train_features = []
		train_targets = []
		test_features = []

		train_dict, test_dict = self.get_data_fp()
		
		# Iterate through train_data
		print("Processing train image data")

		for name in train_dict:
			img_fp = train_dict[name]['image']
			im = imread(img_fp)[...,:3]
			w, h, _ = im.shape
			r = w/h

			# load masks
			masks_fp = train_dict[name]['mask']
			masks = list(map(lambda m: imread(m), masks_fp))

			# flatten masks
			masks = flatten_masks(masks)

			# Turn rgb channels to grayscale
			im = rgb_to_gray(im)

			# Sample-crop and resize nonsquare images and masks
			if r == 1:
				im1 = resizer(im)
				masks = resizer(masks)
				train_features += [im1]
				train_targets += [masks]
			elif r < self.settings.dim_thres_vertical:
				im1, im2, masks1, masks2 = vertical_sample(im, w, h, masks)
				train_features += [im1, im2]
				train_targets += [masks1, masks2]
			elif r > self.settings.dim_thres_horizontal:
				im1, im2, masks1, masks2 = horizontal_sample(im, w, h, masks)
				train_features += [im1, im2]
				train_targets += [masks1, masks2]
			else:
				im1, masks1 = random_sample(im, w, h, masks)
				train_features += [im1]
				train_targets += [masks1]

		# Iterate through test_features
		print("Processing test image data")

		for name in test_dict:
			img_fp = test_dict[name]['image']
			im = imread(img_fp)[...,:3]
			w, h, _ = im.shape
			r = w/h

			# Turn rgb channels to grayscale
			im = rgb_to_gray(im)

			# Sample-crop and resize nonsquare images and masks
			if r == 1:
				test_features.append(im)
			elif r < self.settings.dim_thres_vertical:
				im1, im2 = vertical_sample(im, w, h)
				test_features += [im1, im2]
			elif r > self.settings.dim_thres_horizontal:
				im1, im2 = horizontal_sample(im, w, h)
				test_features += [im1, im2]
			else:
				im1 = random_sample(im, w, h)
				test_features += [im1]

		print('Pickling data to file')
		
		with open(self.settings.train_features_pkl, 'wb') as f:
			pickle.dump(train_features, f)
		with open(self.settings.train_targets_pkl, 'wb') as f:
			pickle.dump(train_targets, f)
		with open(self.settings.test_features_pkl, 'wb') as f:
			pickle.dump(test_features, f)

		print('Done processing image data!')

	def load_pkl_data(self, fp):
		with open(fp, 'rb') as f:
			return pickle.load(f)

	def train_validation_split(self):
		pass

	def visualize_image(self, img, mask=None):
		pass

	def transform_images(self, images):
		pass

	def augment_images(self, images):
		pass

# image_class = NucleiImages()
# image_data.process_save_data()
# test_fp = image_class.settings.train_targets_pkl
# data = image_class.load_pkl_data(test_fp)

# print(np.expand_dims(data[0], axis=-1))


# imshow(train_im[92], cmap='Greys')
# plt.hist(test[92].ravel(), 256, [0,256])
