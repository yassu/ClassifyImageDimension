#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os.path
import pickle
import numpy as np
import sys
from skimage import io
from skimage.feature import hog
from sklearn import datasets, metrics, ensemble


def get_classifier(data, target):
    classifier = ensemble.RandomForestClassifier(
        n_estimators=20,
        max_depth=3,
        criterion='gini')
    classifier.fit(data, target)
    return classifier


def load_images(path):
    filenames = glob.glob(os.path.join(path, '*/*.png'))
    hogs = np.ndarray((len(filenames), 57600), dtype=np.float)
    labels = np.ndarray(len(filenames), dtype=np.int)

    for j, filename in enumerate(filenames):
        image = io.imread(filename, as_grey=True)
        hogs[j] = hog(
            image,
            orientations=9,
            pixels_per_cell=(5, 5),
            cells_per_block=(5, 5),
            block_norm='L2-Hys'
        )
        labels[j] = int([os.path.split(os.path.dirname(filename))[-1]][0][0])

    return datasets.base.Bunch(
        data=hogs,
        target=labels.astype(np.int),
        target_names=np.arange(2),
        DESCR=None)


def show_statics(target, predicted):
    print("Confusion matrix:{}".format(
        metrics.confusion_matrix(test.target, predicted)))
    print("Accuracy:{}".format(
        metrics.accuracy_score(test.target, predicted)))


if __name__ == '__main__':
    train_path, test_path = sys.argv[1], sys.argv[2]

    print('Start to load images')
    train = load_images(train_path)
    classifier = get_classifier(train.data, train.target)

    print('Start to test images')
    test = load_images(test_path)
    print('Start to predict')
    predicted = classifier.predict(test.data)

    show_statics(test.target, predicted)
