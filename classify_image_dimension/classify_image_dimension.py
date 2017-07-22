#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os.path
import pickle
import numpy as np
import sys
from skimage import io
from skimage.feature import hog
from sklearn import datasets, metrics, ensemble


DEFAULT_PICKLE_FILENAME = os.path.expanduser(
    '~/.image_dimension_classifier.pickle')
SYSTEM_PICKLE_FILENAME = os.path.join(
    os.path.dirname(__file__),
    '..', 'data', 'image_dimension_classifier.pickle')


def get_classifier(data, target):
    classifier = ensemble.RandomForestClassifier(
        n_estimators=20,
        max_depth=3,
        criterion='gini')
    classifier.fit(data, target)
    return classifier


def get_maked_classifiler(default_pickle_filename=DEFAULT_PICKLE_FILENAME):
    if os.path.exists(DEFAULT_PICKLE_FILENAME):
        pickle_filename = DEFAULT_PICKLE_FILENAME
    else:
        pickle_filename = SYSTEM_PICKLE_FILENAME

    print(pickle_filename)
    with open(pickle_filename, mode='rb') as f:
        # TODO: pickle_filenameが不正な形式のときのエラー処理
        return pickle.load(f)


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
        metrics.confusion_matrix(target, predicted)))
    print("Accuracy:{}".format(
        metrics.accuracy_score(target, predicted)))


def get_parser():
    parser = argparse.ArgumentParser(description='classify image dimension')
    parser.add_argument(
        '--predict', '-p',
        action='store',
        dest='predicted_path',
        help='predict')

    return parser


def main(args):
    # print('Start to load images')
    # train = load_images(train_path)
    # classifier = get_classifier(train.data, train.target)
    #
    # with open(
    #     os.path.expanduser(DEFAULT_PICKLE_FILENAME),
    #         mode='wb') as f:
    #     pickle.dump(classifier, f)
    #

    if args.predicted_path:
        predicted_path = args.predicted_path

        classifier = get_maked_classifiler()
        print('Start to predicted images')
        # TODO: cropしたりpngにしたりする処理を入れる
        test = load_images(predicted_path)
        print('Start to predict')
        predicted = classifier.predict(test.data)

        show_statics(test.target, predicted)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
