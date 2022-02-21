from __future__ import absolute_import
from __future__ import print_function
import os
import argparse
import numpy as np
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

# Set random seed
np.random.seed(123)


def other_class(n_classes, current_class):
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


def get_lr_scheduler(dataset):
    if dataset in ['mnist', 'MNIST']:
        def scheduler(epoch):
            if epoch > 30:
                return 0.001
            elif epoch > 10:
                return 0.01
            else:
                return 0.1

        return LearningRateScheduler(scheduler)


##################### Dataset Generation Code ############################################

NUM_CLASSES = {'mnist': 10}


def get_data(dataset, noise_ratio, asym, balanced, random_shuffle):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if balanced:
        X_train_reshaped = X_train.reshape(-1, 784)
        ros = RandomOverSampler()
        X_balanced, Y_balanced = ros.fit_resample(X_train_reshaped, y_train)
        X_train = X_balanced.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        y_train = Y_balanced
        y_train_clean = np.copy(Y_balanced)
        Y_clean_train = np.copy(Y_balanced)
        Y_clean_test = np.copy(y_test)
    else:
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        y_train = y_train
        y_train_clean = np.copy(y_train)
        Y_clean_train = np.copy(y_train)
        Y_clean_test = np.copy(y_test)
    if noise_ratio > 0:
        if asym:
            data_file = "asym_%s_train_labels_%s.npy" % (dataset, noise_ratio)
        else:
            data_file = "%s_train_labels_%s.npy" % (dataset, noise_ratio)
        if os.path.isfile(data_file):
            y_train = np.load(data_file)

        else:
            if asym:
                # 1 < - 7, 2 -> 7, 3 -> 8, 5 <-> 6
                source_class = [7, 2, 3, 5, 6]
                target_class = [1, 7, 8, 6, 5]
                for s, t in zip(source_class, target_class):
                    cls_idx = np.where(y_train_clean == s)[0]
                    n_noisy = int(noise_ratio * cls_idx.shape[0] / 100)
                    noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                    y_train[noisy_sample_index] = t
            else:
                n_samples = y_train.shape[0]
                n_noisy = int(noise_ratio * n_samples / 100)
                class_index = [np.where(y_train_clean == i)[0] for i in range(NUM_CLASSES[dataset])]
                class_noisy = int(n_noisy / NUM_CLASSES[dataset])

                noisy_idx = []
                for d in range(NUM_CLASSES[dataset]):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)

                for i in noisy_idx:
                    y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])

            np.save(data_file, y_train)

        print("Print noisy label generation statistics:")
        for i in range(NUM_CLASSES[dataset]):
            n_noisy = np.sum(y_train == i)
            print("Noisy class %s, has %s samples." % (i, n_noisy))

    if random_shuffle:
        idx_perm = np.random.permutation(X_train.shape[0])
        X_train, y_train, y_train_clean = X_train[idx_perm], y_train[idx_perm], y_train_clean[idx_perm]

    # one-hot-encode the labels
    y_train_clean = np_utils.to_categorical(y_train_clean, NUM_CLASSES[dataset])
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES[dataset])
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES[dataset])

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test", y_test.shape)
    Y_train = np.argmax(y_train, axis=1)
    clean_selected = np.argwhere(Y_train == Y_clean_train).reshape((-1,))
    noisy_selected = np.argwhere(Y_train != Y_clean_train).reshape((-1,))
    print("#correct labels: %s, #incorrect labels: %s" % (len(clean_selected), len(noisy_selected)))

    return X_train, y_train, y_train_clean, X_test, y_test


################## Loss Functions #####################################################################

def cross_entropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return alpha * tf.reduce_mean(
            -tf.reduce_sum(y_true_1 * tf.math.log(y_pred_1), axis=-1)) + beta * tf.reduce_mean(
            -tf.reduce_sum(y_pred_2 * tf.math.log(y_true_2), axis=-1))

    return loss


def normsym(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred
        y_true_2 = y_true
        y_pred_2 = y_pred
        # y_pred_1 = tf.nn.log_softmax(y_pred)

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        normalizor = 1 / 4 * (10 - 1)

        ce = tf.reduce_mean(
            -tf.reduce_sum(y_true_1 * tf.math.log(y_pred_1), axis=-1))  # /(-tf.reduce_sum(y_pred_1, axis=-1))
        rce = tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.math.log(y_true_2), axis=-1))
        nce = alpha * ce
        nrce = beta * normalizor * rce
        return nce + nrce

    return loss


################## Convoultional Model ###############################

def get_model(dataset='mnist', input_tensor=None, input_shape=None, num_classes=10):
    """
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    input_shape: optional shape tuple
    :return: The model; a Keras 'Model' instance.
    """
    assert dataset in ['mnist']

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_shape):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", name='conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    x = Flatten()(x)

    x = Dense(128, kernel_initializer="he_normal", name='fc1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='lid')(x)
    # x = Dropout(0.2)(x)

    x = Dense(num_classes, kernel_initializer="he_normal")(x)
    x = Activation(tf.nn.softmax)(x)

    model = Model(img_input, x)

    return model


########## Log train/val loss and acc into file for later plots ########################################

class LoggerCallback(Callback):
    """
    Log train/val loss and acc into file for later plots.
    """

    def __init__(self, model, X_train, y_train, y_train_clean, X_test, y_test, dataset,
                 model_name, noise_ratio, asym, epochs, alpha, beta):
        super(LoggerCallback, self).__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_clean = y_train_clean
        self.X_test = X_test
        self.y_test = y_test
        self.n_class = y_train.shape[1]
        self.dataset = dataset
        self.model_name = model_name
        self.noise_ratio = noise_ratio
        self.asym = asym
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta

        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.train_loss_class = [None] * self.n_class
        self.train_acc_class = [None] * self.n_class

        # the followings are used to estimate LID
        self.lid_k = 20
        self.lid_subset = 128
        self.lids = []

        # complexity - Critical Sample Ratio (csr)
        self.csr_subset = 500
        self.csr_batchsize = 100
        self.csrs = []

    def on_epoch_end(self, epoch, logs={}):
        tr_acc = logs.get('acc')
        tr_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')

        self.train_loss.append(tr_loss)
        self.test_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.test_acc.append(val_acc)

        # print('ALL acc:', self.test_acc)

        if self.asym:
            file_name = 'log/asym_loss_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss))))
            file_name = 'log/asym_acc_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.stack((np.array(self.train_acc), np.array(self.test_acc))))
            file_name = 'log/asym_class_loss_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.array(self.train_loss_class))
            file_name = 'log/asym_class_acc_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.array(self.train_acc_class))
        else:
            file_name = 'log/loss_%s_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio, self.alpha)
            np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss))))
            file_name = 'log/acc_%s_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio, self.alpha)
            np.save(file_name, np.stack((np.array(self.train_acc), np.array(self.test_acc))))

        return


class SGDLearningRateTracker(Callback):
    def __init__(self, model):
        super(SGDLearningRateTracker, self).__init__()
        self.model = model

    def on_epoch_begin(self, epoch, logs={}):
        init_lr = float(K.get_value(self.model.optimizer.lr))
        decay = float(K.get_value(self.model.optimizer.decay))
        iterations = float(K.get_value(self.model.optimizer.iterations))
        lr = init_lr * (1. / (1. + decay * iterations))
        print('init lr: %.4f, current lr: %.4f, decay: %.4f, iterations: %s' % (init_lr, lr, decay, iterations))


############# Metrics #######################################

def auc(y_true, y_pred):
    return metrics.roc_auc_score(K.eval(y_true), K.eval(y_pred))


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


############# Fit Predict and Plot #############################################

# prepare folders
folders = ['data', 'model', 'log']
for folder in folders:
    path = os.path.join('./', folder)
    if not os.path.exists(path):
        os.makedirs(path)


def train(model_name='sl', batch_size=128, epochs=50, noise_ratio=0, noise_type='sym', dataset_type='imbal',
          alpha=1.0, beta=1.0):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param model_name:
    :param batch_size:
    :param epochs:
    :param noise_ratio:
    :param noise_type:
    :param dataset_type:
    :param alpha:
    :param beta:
    :return:
    """
    dataset = 'mnist'

    print(
        'Dataset: %s, model: %s, batch: %s, epochs: %s, noise ratio: %s%%, noise_type: %s, dataset_type: %s, alpha: %s, beta: %s' %
        (dataset, model_name, batch_size, epochs, noise_ratio, noise_type, dataset_type, alpha, beta))

    # load data
    # X_train, y_train, y_train_clean, X_test, y_test = get_data(dataset, noise_ratio=noise_ratio, asym=asym,balanced=balanced, random_shuffle=False)

    if noise_type in ['sym', 'Sym', 'SYM']:
        if dataset_type in ['bal', 'Bal', 'BAL']:
            X_train, y_train, y_train_clean, X_test, y_test = get_data(dataset, noise_ratio=noise_ratio, asym=False,
                                                                       balanced=True, random_shuffle=False)
        elif dataset_type in ['imbal', 'Imbal', 'IMBAL']:
            X_train, y_train, y_train_clean, X_test, y_test = get_data(dataset, noise_ratio=noise_ratio, asym=False,
                                                                       balanced=False, random_shuffle=False)
    elif noise_type in ['asym', 'Asym', 'ASYM']:
        if dataset_type in ['bal', 'Bal', 'BAL']:
            X_train, y_train, y_train_clean, X_test, y_test = get_data(dataset, noise_ratio=noise_ratio, asym=True,
                                                                       balanced=True, random_shuffle=False)
        elif dataset_type in ['imbal', 'Imbal', 'IMBAL']:
            X_train, y_train, y_train_clean, X_test, y_test = get_data(dataset, noise_ratio=noise_ratio, asym=True,
                                                                       balanced=False, random_shuffle=False)
    else:
        print("Unknown arguments! Retry...")
        exit(0)

    n_images = X_train.shape[0]
    image_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    print("n_images", n_images, "num_classes", num_classes, "image_shape:", image_shape)

    # load model
    model = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=num_classes)
    model.summary()

    optimizer = SGD(lr=0.1, decay=1e-4, momentum=0.9)

    # create loss
    if model_name == 'ce':
        loss = cross_entropy
    elif model_name == 'sl':
        loss = symmetric_cross_entropy(alpha, beta)
    elif model_name == 'normsym':
        loss = normsym(alpha, beta)
    else:
        print("Model %s is unimplemented!" % model_name)
        exit(0)

    # model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy', f1_m, precision_m, recall_m]
    )

    model_save_file = "model/%s_%s_%s_%s_%s.{epoch:02d}.hdf5" % (
        model_name, dataset, dataset_type, noise_ratio, noise_type)

    ## do real-time updates using callbakcs
    callbacks = []
    cp_callback = ModelCheckpoint(model_save_file,
                                  monitor='val_loss',
                                  verbose=0,
                                  save_best_only=False,
                                  save_weights_only=True,
                                  period=1)
    callbacks.append(cp_callback)

    # es = EarlyStopping(monitor='accuracy', mode='max', verbose=1,restore_best_weights = True, patience=20)
    # callbacks.append(es)

    # learning rate scheduler
    lr_scheduler = get_lr_scheduler(dataset)
    callbacks.append(lr_scheduler)
    callbacks.append(SGDLearningRateTracker(model))

    # acc, loss, lid
    if noise_type in ['asym', 'Asym', 'ASYM']:
        log_callback = LoggerCallback(model, X_train, y_train, y_train_clean, X_test, y_test, dataset, model_name,
                                      noise_ratio, True, epochs, alpha, beta)
    else:
        log_callback = LoggerCallback(model, X_train, y_train, y_train_clean, X_test, y_test, dataset, model_name,
                                      noise_ratio, False, epochs, alpha, beta)

    callbacks.append(log_callback)

    # data augmentation and split
    datagen = ImageDataGenerator(validation_split=0.2)
    train_dataset = datagen.flow(X_train, y_train, batch_size=batch_size, subset='training')
    val_dataset = datagen.flow(X_train, y_train, batch_size=batch_size, subset='validation')
    datagen.fit(X_train)

    # train the model
    history = model.fit(train_dataset,
                        steps_per_epoch=len(train_dataset) // batch_size, epochs=epochs,
                        validation_data=val_dataset,
                        verbose=1,
                        callbacks=callbacks
                        )

    # Graphing our training and validation

    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    print("Loss incurred on test data is {}".format(loss))
    print("Accuracy acquired on test data is {}".format(accuracy))
    print("F1 score on test data is {}".format(f1_score))
    print("Precision acquired on test data is {}".format(precision))
    print("Recall acquired on test data is {}".format(recall))
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    print("Plotting confusion matrix")

    fig = plt.figure(figsize=(10, 10))  # Set Figure

    y_pred = model.predict(X_test)  # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    y_pred_de = np.argmax(y_pred, 1)  # Decode Predicted labels
    y_test_de = np.argmax(y_test, 1)  # Decode labels

    mat = confusion_matrix(y_test_de, y_pred_de)  # Confusion matrix
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.show()

    print("ROC AUC curve")

    predict_label1 = np.argmax(y_pred, axis=-1)
    true_label1 = np.argmax(y_test, axis=-1)
    y = np.array(true_label1)
    scores = np.array(predict_label1)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=9)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


def main(args):
    train(model_name=args.model_name, batch_size=args.batch_size, epochs=args.epochs, noise_ratio=args.noise_ratio,
          noise_type=args.noise_type, dataset_type=args.dataset_type, alpha=args.alpha, beta=args.beta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model_name',
        help="Model name: 'ce', 'sl', 'normsym' ",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-r', '--noise_ratio',
        help="The percentage of noisy labels [0, 100].",
        required=False, type=int
    )
    parser.add_argument(
        '-t', '--noise_type',
        help="Noise type: 'sym', 'asym' ",
        required=False, type=str
    )
    parser.add_argument(
        '-d', '--dataset_type',
        help="Type of dataset: 'bal', 'imbal' ",
        required=False, type=str
    )
    parser.add_argument(
        '-alpha', '--alpha',
        help="Alpha parameter (float)",
        required=True, type=float
    )
    parser.add_argument(
        '-beta', '--beta',
        help="Beta parameter (float)",
        required=True, type=float
    )
    parser.set_defaults(epochs=150)
    parser.set_defaults(batch_size=128)
    parser.set_defaults(noise_ratio=0)
    parser.set_defaults(noise_type='asym')
    parser.set_defaults(dataset_type='imbal')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parser.parse_args()

    # If not running from console use this instead
    # args = parser.parse_args(['-m', 'normsym',
    #                           '-b', '128', '-e', '150',
    #                           '-r', '40', '-t', 'sym', '-d', 'bal',
    #                           '-alpha', '0.1', '-beta', '1.0'])


    main(args)
    K.clear_session()
