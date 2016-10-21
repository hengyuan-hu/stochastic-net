import keras

import resnet_cifar10
import data_cifar10
from keras.callbacks import TensorBoard
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot

if __name__ == '__main__':
    (train_xs, train_ys), (test_xs, test_ys) = data_cifar10.load_cifar10()

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    model, model_name = resnet_cifar10.resnet_cifar10(repetations=3)
    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    plot(model,to_file='model.png')

    history = model.fit_generator(datagen.flow(train_xs, train_ys, batch_size=128),
                        samples_per_epoch = train_xs.shape[0],
                        nb_epoch=200,
                        validation_data=(test_xs, test_ys),
                        callbacks = [TensorBoard()])

    model.save_weights('model.h5')
