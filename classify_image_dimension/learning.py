#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os.path
import numpy as np
import sys
from skimage import io
from sklearn import datasets, metrics, tree

IMAGE_SIZE = 100
COLOR_BYTE = 3


def load_images(path):
    filenames = glob.glob(os.path.join(path, '*/*.png'))
    images = np.ndarray(
        (
            len(filenames), IMAGE_SIZE, IMAGE_SIZE,
            COLOR_BYTE), dtype=np.int)
    labels = np.ndarray(len(filenames), dtype=np.int)

    for j, file in enumerate(filenames):
        image = io.imread(file)
        images[j] = image

        labels[j] = int([os.path.split(os.path.dirname(file))[-1]])

    flat_data = images.reshape(-1, IMAGE_SIZE * IMAGE_SIZE * COLOR_BYTE)
    images = flat_data.view()
    return datasets.base.Bunch(
        data=flat_data,
        target=labels.astype(np.int),
        target_names=np.arange(2),
        images=images,
        DESCR=None)


if __name__ == '__main__':
    train_path, test_path = sys.argv[1], sys.argv[2]

    train = load_images(train_path)

    # TODO: criterion='entropy' or 'gini', max_depth=3 or 5のうち
    # もっとも上手くいったのでこの値にした.(Accuracy: 65.5%)
    # 後でハイパーパラメタ用のデータ・セットを作ってテストし直したい
    classifier = tree.DecisionTreeClassifier(max_depth=5, criterion='entropy')
    classifier.fit(train.data, train.target)

    test = load_images(test_path)
    predicted = classifier.predict(test.data)

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(
        test.target, predicted))
    print("Accuracy:\n%s" % metrics.accuracy_score(test.target, predicted))
