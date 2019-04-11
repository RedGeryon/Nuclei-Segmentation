import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Cropping2D, concatenate, UpSampling2D
from keras import backend as K
import tensorflow as tf



class BasicUnet():
	def __init__(self, img_dim = None,  n_classes = 1):
		self.img_dim = img_dim
		self.n_classes = n_classes
		self.model = self.Unet()

	def DownBlock(self, x, filters):
		x = Conv2D(filters=filters, kernel_size=3, padding='same',
				activation='relu', kernel_initializer='he_normal')(x)
		x = Conv2D(filters=filters, kernel_size=3, padding='same',
				activation='relu', kernel_initializer='he_normal')(x)

		return MaxPooling2D(pool_size=(2,2))(x), x

	def UpBlock(self, x, cross, filters):
		x = UpSampling2D(size=(2,2))(x)
		x = Conv2D(filters=filters, kernel_size=(2,2), padding='same')(x)
		x = concatenate([x, cross], axis=3)
		x = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
		x = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
		return x

	def Unet(self):
		# Downward path
		inputs = Input(self.img_dim)
		x, cross1 = self.DownBlock(inputs, 64)
		x, cross2 = self.DownBlock(x, 128)
		x, cross3 = self.DownBlock(x, 256)
		x, cross4 = self.DownBlock(x, 512)
		_, x = self.DownBlock(x, 1024)

		# Upward path
		x = self.UpBlock(x, cross4, 512)
		x = self.UpBlock(x, cross3, 256)
		x = self.UpBlock(x, cross2, 128)
		x = self.UpBlock(x, cross1, 64)
		x = Conv2D(filters=self.n_classes, kernel_size=(1,1), padding='same', activation='sigmoid')(x)
		return Model(inputs, outputs = x)

	def train(self):
		pass

	def predict(self):
		pass

	def score(self):
		pass
