import cPickle

import tensorflow as tf
import keras
from tensorflow.python.platform import app, flags

from utils_tf import tf_model_train, tf_model_eval, batch_eval
from misc import get_session, save_model, load_model
import resnet_cifar10
import data_cifar10


FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', './train', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'cifar10.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 160, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 32, 'Input row dimension')
flags.DEFINE_integer('img_cols', 32, 'Input column dimension')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def adam_pretrain(model, model_name, train_xs, train_ys, num_epoch, test_xs, test_ys):
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_xs, train_ys, batch_size=128, nb_epoch=num_epoch,
              validation_data=(test_xs, test_ys), shuffle=True)
    model_name = '%s_adam_pretrain' % model_name
    save_model(model, model_name)
    model = load_model(model_name)
    return model


def main():
    if keras.backend.image_dim_ordering() != 'th':
        keras.backend.set_image_dim_ordering('th')
        print "INFO: temporarily set 'image_dim_ordering' to 'th'"

    sess = get_session()
    keras.backend.set_session(sess)

    (train_xs, train_ys), (test_xs, test_ys) = data_cifar10.load_cifar10()
    print 'Loaded cifar10 data'

    x = tf.placeholder(tf.float32, shape=(None, 3, 32, 32))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model, model_name = resnet_cifar10.resnet_cifar10(repetations=3)

    predictions = model(x)
    tf_model_train(sess, x, y, predictions, train_xs, train_ys, test_xs, test_ys,
                   data_augmentor=data_cifar10.augment_batch)

    save_model(model, model_name)


if __name__ == '__main__':
    main()
