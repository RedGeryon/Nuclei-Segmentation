import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Cropping2D, concatenate, UpSampling2D, Lambda, Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from augmentdata import augment
import tensorflow as tf
import custplotlosses

class BasicUnet():
    def __init__(self, img_dim = None,  n_classes = 1, cost='bce', class_weight=None, contour_weight = 0.2):
        self.img_dim = img_dim
        self.n_classes = n_classes
        if cost == 'bce_dice':
            self.cost = self.bce_dice
        elif cost == 'weighted_bce_dice':
            self.cost = self.weighted_bce_dice
        else:
            self.cost = self.binary_crossentropy
            
        self.class_weight = class_weight
        self.contour_weight = contour_weight
        self.model = self.Unet()
        # Define callbacks: earlystop, checkpoint, and custom plotter
        self.earlystop = EarlyStopping(patience=5, verbose=1)
        self.checkpoint = ModelCheckpoint('nuclei_segment.h5', verbose=1, save_best_only=True)
        self.custplot = custplotlosses.PlotLosses()
        

    def DownBlock(self, x, filters):
        x = Conv2D(filters=filters, kernel_size=3, padding='same',
                activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(filters=filters, kernel_size=3, padding='same',
                activation='relu', kernel_initializer='he_normal')(x)

        return MaxPooling2D(pool_size=(2,2))(x), x

    def UpBlock(self, x, cross, filters):
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, cross], axis=3)
        x = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        return x

    def Unet(self):
        # Downward path
        inputs = Input(self.img_dim)
        x = Lambda(lambda x: x / 255.) (inputs)
        x, cross1 = self.DownBlock(inputs, 8)
        x, cross2 = self.DownBlock(x, 16)
        x, cross3 = self.DownBlock(x, 32)
        x, cross4 = self.DownBlock(x, 64)
        _, x = self.DownBlock(x, 128)

        # Upward path
        x = self.UpBlock(x, cross4, 64)
        x = self.UpBlock(x, cross3, 32)
        x = self.UpBlock(x, cross2, 16)
        x = self.UpBlock(x, cross1, 8)
        x = Conv2D(filters=self.n_classes, kernel_size=(1,1), padding='same', activation='sigmoid')(x)

        model = Model(inputs, outputs = x)
        model.compile(optimizer='adam', loss=self.cost , metrics=[self.mean_iou])
        return model

    def mean_iou(self, y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)
    
    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

    def binary_crossentropy(self, y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred))

    def dice_coef_loss(self, y_true, y_pred):
        return -self.dice_coef(y_true, y_pred)
    
    def bce_dice(self, y_true, y_pred):
        return self.dice_coef_loss(y_true, y_pred) + self.binary_crossentropy(y_true, y_pred)
    
    def weighted_bce_dice(self, y_true, y_pred):
        '''Weighted loss for multichannel classification'''
        # Contour weight is from 0-1 (fraction of total weight)
        # 1st channel is original mask, 2nd channel is contour mask
        mask_bce_dice = self.bce_dice(y_true[...,0], y_pred[...,0])
        contour_bce_dice = self.bce_dice(y_true[...,1], y_pred[...,1])
        return self.contour_weight*contour_bce_dice + (1-self.contour_weight)*mask_bce_dice

    def train(self, X_train, Y_train, val_split = .1):
        self.model.fit(X_train, Y_train, validation_split= val_split, shuffle=True, batch_size=8, epochs=100, 
                    callbacks=[self.earlystop, self.checkpoint, self.custplot], class_weight=self.class_weight)
        
    def train_augment(self, X_train, Y_train, val_split = .1):
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=val_split)
        
        train_gen, train_samples = augment(X_train, Y_train)
        val_gen, val_samples = augment(X_test, Y_test, test=True)
        
        self.model.fit_generator(train_gen, validation_data=val_gen, validation_steps=val_samples,
                    steps_per_epoch=train_samples, epochs=100, callbacks=[self.earlystop, self.checkpoint, self.custplot],  class_weight=self.class_weight)

    def predict(self, data):
        return self.model.predict(data)