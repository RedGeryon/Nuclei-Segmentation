from tensorflow.keras.preprocessing.image import ImageDataGenerator 

def augment(X_train, Y_train, test=False):
    
    if test:
        traingen = ImageDataGenerator()
        maskgen = ImageDataGenerator()
        
        traingen.fit(X_train, augment=True, seed=10)
        maskgen.fit(X_train, augment=True, seed=10)
        
        test = traingen.flow(X_train, batch_size=16, shuffle=False, seed=10)
        mask = maskgen.flow(Y_train, batch_size=16, shuffle=False, seed=10)
        
        # Returns generator and batch size
        return zip(test, mask), len(X_train)/16
    else:
        # Initialize same transformations across feature and mask
        transformations = dict(width_shift_range = 0.2,
                                height_shift_range = 0.2,
                                horizontal_flip = True,
                                vertical_flip = True,
                                shear_range=0.2,
                                zoom_range=0.2,
                                fill_mode='reflect')

        traingen = ImageDataGenerator (**transformations)
        maskgen = ImageDataGenerator (**transformations)

        traingen.fit(X_train, augment=True, seed=10)
        maskgen.fit(Y_train, augment=True, seed=10)
        train_aug = traingen.flow(X_train, batch_size=16, shuffle=True, seed=10)
        mask_aug = maskgen.flow(Y_train, batch_size=16, shuffle=True, seed=10)

        # Return a generator and batch size
        return zip(train_aug, mask_aug), len(X_train)/16