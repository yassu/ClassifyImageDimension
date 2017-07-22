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


if __name__ == '__main__':
    train_path, test_path = sys.argv[1], sys.argv[2]

    train = load_images(train_path)

    # TODO: criterion='entropy' or 'gini', max_depth=3 or 5のうち
    # もっとも上手くいったのでこの値にした.(Accuracy: 65.5%)
    # 後でハイパーパラメタ用のデータ・セットを作ってテストし直したい
    classifier = ensemble.RandomForestClassifier(
        n_estimators=20,
        max_depth=3,
        criterion='gini')
    classifier.fit(train.data, train.target)

    test = load_images(test_path)
    predicted = classifier.predict(test.data)

    print("Confusion matrix:{}".format(
        metrics.confusion_matrix(test.target, predicted)))
    print("Accuracy:{}".format(
        metrics.accuracy_score(test.target, predicted)))
