from __future__ import absolute_import
from __future__ import print_function
import os
import argparse
from keras.utils import np_utils
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from keras.datasets import mnist
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

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


NUM_CLASSES = {'mnist': 10}


def get_data(dataset, noise_ratio, asym, balanced, random_shuffle):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if balanced:
        X_train_reshaped = X_train.reshape(-1, 784)
        ros = RandomOverSampler()
        X_balanced, Y_balanced = ros.fit_resample(X_train_reshaped, y_train)
        # print(X_balanced.shape[0] - X_train_reshaped.shape[0], 'new random picked points')
        X_train = X_balanced.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        # X_train = X_train / 255.0
        # X_test = X_test / 255.0
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        y_train = Y_balanced
        y_train_clean = np.copy(Y_balanced)
        Y_clean_train = np.copy(Y_balanced)
        Y_clean_test = np.copy(y_test)
    else:
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        # X_train = X_train / 255.0
        # X_test = X_test / 255.0
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
        #################################################################
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


def mlmodel(noise_ratio=40, noise_type='sym', dataset_type='imbal'):
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
    X_train_std = StandardScaler().fit_transform(X_train_flattened)
    X_test_std = StandardScaler().fit_transform(X_test_flattened)

    #######Calculating eigen values and eigenvectors manually #####################
    mean_vec = np.mean(X_train_std, axis=0)
    cov_mat = np.cov(X_train_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # Create a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # Sort the eigenvalue, eigenvector pair from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    # Calculation of Explained Variance from the eigenvalues
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]  # Individual ex"plained variance
    cum_var_exp = np.cumsum(var_exp)  # Cumulative explained variance

    # Using plotly to visualise individual explained variance and cummulative explained variance
    trace1 = go.Scatter(
        x=list(range(784)),
        y=cum_var_exp,
        mode='lines+markers',
        name="'Cumulative Explained Variance'",

        line=dict(
            shape='spline',
            color='goldenrod'
        )
    )
    trace2 = go.Scatter(
        x=list(range(784)),
        y=var_exp,
        mode='lines+markers',
        name="'Individual Explained Variance'",

        line=dict(
            shape='linear',
            color='black'
        )
    )
    fig = tls.make_subplots(insets=[{'cell': (1, 1), 'l': 0.7, 'b': 0.5}],
                            print_grid=True)

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)

    fig.layout.title = 'explained Variance plots'
    fig.layout.xaxis = dict(range=[0, 800], title='Feature columns')
    fig.layout.yaxis = dict(range=[0, 100], title='explained variance')

    py.iplot(fig, filename='inset example')

    # We see that nearly 90% of the explained variance can be explained by 200 features

    ####################Plotting cumulative variance for the prinicpal components####################
    pca = PCA(200)
    pca_full = pca.fit(X_train_std)

    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlabel('# of components')
    plt.ylabel('Cumulative explained variance')

    ########Fittting with PCA#########################
    pca = PCA(n_components=50)
    X_train_transformed = pca.fit_transform(X_train_std)
    X_test_transformed = pca.transform(X_test_std)

    #############The below codes tries a permuattaion and combination of best k with the best principal componnets to fit the Knn model#########

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_train_transformed, Y_train_new, test_size=0.2,
                                                                        random_state=13)
    components = [15, 25, 35, 45, 50]
    neighbors = [25, 35, 45]

    scores = np.zeros((components[len(components) - 1] + 1, neighbors[len(neighbors) - 1] + 1))

    for component in components:
        for n in neighbors:
            knn = KNeighborsClassifier(n_neighbors=n)
            knn.fit(X_train_pca[:, :component], y_train_pca)
            score = knn.score(X_test_pca[:, :component], y_test_pca)
            # predict = knn.predict(X_test_pca[:,:component])
            scores[component][n] = score

            print('Components = ', component, ', neighbors = ', n, ', Score = ', score)

    ##### Fitting with the best k and best principal components###############

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train_pca[:, :35], y_train_pca)
    predict_labels = knn.predict(X_test_transformed[:, :35])
    pred_prob = knn.predict_proba(X_test_transformed[:, :35])

    ###########Accuracy score and confuion matrix#######################
    print("accuracy:", metrics.accuracy_score(y_true=Y_test_new, y_pred=predict_labels), "\n")
    print(metrics.confusion_matrix(y_true=Y_test_new, y_pred=predict_labels))
    print(precision_recall_fscore_support(y_true=Y_test_new, y_pred=predict_labels, average='micro'))

    ######################Precision Recall  curve##########################################################
    precision = dict()
    recall = dict()
    n_classes = 10
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            pred_prob[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()

    ##################ROC curve and AUC arear prediction for each of them ###############################################
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i],
                                      pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label="ROC curve of class {0} (area = {1:0.4f})".format(i, roc_auc[i]))

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.show()

    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8',
                    'class 9']
    print(classification_report(y_true=Y_test_new, y_pred=predict_labels, target_names=target_names))


def main(args):
    mlmodel(noise_ratio=args.noise_ratio, noise_type=args.noise_type, dataset_type=args.dataset_type)


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
    args = parser.parse_args()

    #args = parser.parse_args(['-r', '40', '-t', 'sym', '-d', 'bal'])

    main(args)
