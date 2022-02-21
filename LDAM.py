from __future__ import absolute_import
from __future__ import print_function
import os
import pandas as pd
import tensorflow as tf
import argparse
from keras.utils import np_utils
from imblearn.over_sampling import RandomOverSampler
from keras.datasets import mnist
import torch
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import History
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn import metrics

# Set random seed
np.random.seed(123)
NUM_CLASSES = {'mnist': 10}


def other_class(n_classes, current_class):
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class



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

    # generate random noisy labels
    if noise_ratio > 0:
        if asym and balanced:
            data_file = "asym_bal_%s_train_labels_%s.npy" % (dataset, noise_ratio)
        elif asym:
            data_file = "asym_imbal_%s_train_labels_%s.npy" % (dataset, noise_ratio)
        elif balanced:
            data_file = "sym_bal_%s_train_labels_%s.npy" % (dataset, noise_ratio)
        else:
            data_file = "sym_imbal_%s_train_labels_%s.npy" % (dataset, noise_ratio)
        
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

        # print statistics
        print("Print noisy label generation statistics:")
        for i in range(NUM_CLASSES[dataset]):
            n_noisy = np.sum(y_train == i)
            print("Noisy class %s, has %s samples." % (i, n_noisy))

    if random_shuffle:
        # random shuffle
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
    return X_train, y_train, X_test, y_test

### Resnet Model

def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu',
                      padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x


def res_net_model(n_classes=10, num_res_net_blocks=1):

    inputs1 = keras.Input(shape=(28, 28, 1))
    x = layers.BatchNormalization()(inputs1)

    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    for _ in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)      #resnet_block
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes)(x)  # no softmax

    return keras.Model([inputs1], outputs)

# Loss Function
class LDAMLoss():

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = tf.convert_to_tensor(m_list, dtype=tf.float32)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.n_classes = len(cls_num_list)

    def __call__(self, target, x):
        # contrary to pytorch implemenation, our labels are already one hot encoded
        index_float = target
        batch_m = tf.matmul(self.m_list[None, :], tf.transpose(index_float))
        batch_m = tf.reshape(batch_m, (-1, 1))
        x_m = x - batch_m

        # if condition is true, return x_m[index], otherwise return x[index]
        index_bool = tf.cast(index_float, tf.bool)
        output = tf.where(index_bool, x_m, x)

        labels = index_float
        logits = output
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits*self.s)
        return tf.reduce_mean(loss)

def adjust_learning_rate(epoch,lr):
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 180:
        lr = lr * 0.0001
    elif epoch > 160:
        lr = lr * 0.01
    else:
        lr = lr
    return lr

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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def train(noise_ratio=40, noise_type='sym', dataset_type='imbal'):
    dataset = 'mnist'
    if noise_type in ['sym', 'Sym', 'SYM']:
        if dataset_type in ['bal', 'Bal', 'BAL']:
            X_train, Y_train, X_test, Y_test = get_data(dataset, noise_ratio=noise_ratio, asym=False,
                                                        balanced=True, random_shuffle=False)
        elif dataset_type in ['imbal', 'Imbal', 'IMBAL']:
            X_train, Y_train, X_test, Y_test = get_data(dataset, noise_ratio=noise_ratio, asym=False,
                                                        balanced=False, random_shuffle=False)
    elif noise_type in ['asym', 'Asym', 'ASYM']:
        if dataset_type in ['bal', 'Bal', 'BAL']:
            X_train, Y_train, X_test, Y_test = get_data(dataset, noise_ratio=noise_ratio, asym=True,
                                                        balanced=True, random_shuffle=False)
        elif dataset_type in ['imbal', 'Imbal', 'IMBAL']:
            X_train, Y_train, X_test, Y_test = get_data(dataset, noise_ratio=noise_ratio, asym=True,
                                                        balanced=False, random_shuffle=False)
    else:
        print("Unknown arguments! Retry...")
        exit(0)

    X_train_flattened = X_train.reshape(-1, 784)
    Y_train_new = np.argmax(Y_train, axis=1)
    Y_test_new = np.argmax(Y_test, axis=1)
    X_test_flattened = X_test.reshape(-1, 784)

    history = History()
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=5, restore_best_weights=True)
    cb = [history, es]

    model = res_net_model(num_res_net_blocks=1)
    cls_num_list = list(pd.DataFrame(Y_train_new).value_counts().sort_index())

    for epoch in range(0, 50):
        lr = adjust_learning_rate(epoch, 0.01)  #
        idx = epoch // 30
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights)
        obj = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights)  # LDAMLoss
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=obj,
                      metrics=['accuracy', f1_m, precision_m, recall_m])

        model.fit(X_train, Y_train, epochs=epoch, batch_size=150, callbacks=cb, validation_data=(X_test, Y_test))

    # Graphing our training and validation
    acc = history.history['accuracy']
    val_acc = history.history["val_accuracy"]
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

    # Scores
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("F1 score:", score[2])
    print("Precision:", score[3])
    print("Recall:", score[4])

    fig = plt.figure(figsize=(10, 10))  # Set Figure

    y_pred = model.predict(X_test)  # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    y_pred_de = np.argmax(y_pred, 1)  # Decode Predicted labels
    y_test_de = np.argmax(Y_test, 1)  # Decode labels

    mat = confusion_matrix(y_test_de, y_pred_de)  # Confusion matrix

    # Plot Confusion matrix
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.show()

    # Predicting
    prediction = model.predict(X_test)
    predict_label1 = np.argmax(prediction, axis=-1)
    true_label1 = np.argmax(Y_test, axis=-1)

    y = np.array(true_label1)

    scores = np.array(predict_label1)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=9)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def main(args):
    train(noise_ratio=args.noise_ratio, noise_type=args.noise_type, dataset_type=args.dataset_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    parser.set_defaults(noise_ratio=0)
    parser.set_defaults(noise_type='asym')
    parser.set_defaults(dataset_type='imbal')
    # args = parser.parse_args()
    args = parser.parse_args(['-r', '40', '-t', 'sym', '-d', 'bal'])

    main(args)

