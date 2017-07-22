#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os.path
import pickle
import numpy as np
from skimage import io
from skimage.feature import hog
from sklearn import datasets, metrics, ensemble
from tempfile import TemporaryDirectory


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

    with open(pickle_filename, mode='rb') as f:
        # TODO: pickle_filenameが不正な形式のときのエラー処理
        return pickle.load(f)


# TODO: cropしたりpngにしたりする処理を入れる
def load_images(filenames):
    hogs = np.ndarray((len(filenames), 57600), dtype=np.float)
    labels = np.ndarray(len(filenames), dtype=np.int)

    with TemporaryDirectory() as tmpdir_name:
        filenames = crop_filenames(filenames, tmpdir_name)

        for j, filename in enumerate(filenames):
            image = io.imread(filename, as_grey=True)
            hogs[j] = hog(
                image,
                orientations=9,
                pixels_per_cell=(5, 5),
                cells_per_block=(5, 5),
                block_norm='L2-Hys'
            )
            labels[j] = os.path.split(filename)[-1][0]

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


def crop_filenames(filenames, tmpdir_name):
    tmp_filenames = list()
    for filename in filenames:
        basename, ext = os.path.splitext(filename)
        barename = os.path.splitext(os.path.split(filename)[-1])[0]
        tmp_barename = '{}_cropped'.format(barename)
        tmp_filename = os.path.join(tmpdir_name, tmp_barename) + '.png'
        cmd = 'convert {}{} -crop 100x100+0+0 png24:{}'.format(
            basename, ext, tmp_filename)
        os.system(cmd)
        tmp_filenames.append(tmp_filename)
    return tmp_filenames


def get_either_show_statics(filenames):
    if len(filenames) <= 1:
        return False

    for filename in filenames:
        filename = os.path.split(filename)[-1]
        if not(filename.startswith('2d') or filename.startswith('3d')):
            return False
    return True


def get_parser():
    parser = argparse.ArgumentParser(description='classify image dimension')

    parser.add_argument(
        '--train', '-t',
        action='store',
        dest='train_path',
        help='training'
    )

    parser.add_argument(
        '--only-statics',
        action='store_true',
        dest='show_only_statics',
        help='show only statics as result'
    )

    parser.add_argument(
        '--predict', '-p',
        action='store',
        dest='predicted_path',
        help='predict')

    return parser


def main(args):
    if args.train_path:
        train = load_images(args.train_path)
        classifier = get_classifier(train.data, train.target)

        with open(
            os.path.expanduser(DEFAULT_PICKLE_FILENAME),
                mode='wb') as f:
            pickle.dump(classifier, f)

    if args.predicted_path:
        predicted_path = args.predicted_path
        predicted_filenames = glob.glob(
            os.path.join(predicted_path, '*/*.png'))

        classifier = get_maked_classifiler()
        test = load_images(predicted_filenames)
        predicted = classifier.predict(test.data)

        if not args.show_only_statics:
            for filename, predict in zip(predicted_filenames, predicted):
                text = '{:<{}}: {}d'.format(
                    filename,
                    max(map(len, predicted_filenames)) - 1,
                    predict)
                print(text)

        view_statics = get_either_show_statics(predicted_filenames)
        if view_statics:
            show_statics(test.target, predicted)


if __name__ == '__main__':
    main(get_parser().parse_args())
