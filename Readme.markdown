Classify Image Dimension
==========================

This is a program to learn machine learning.

This programming will provide feature that classify 2-dim image or 3-dim image.

## How to use

Most simple how to use is just

```
$ classify_image_dimension {image-path}
2d
```

If you indicate directory as `{image-path}`, you obtain accuracy, precision,
recall and f-measure value.

Other simple how to use is

```
$ classify_image_dimension {training-image-path} {test-image-path}
```

Then, this program trains by `{training-image-path}`, puts data of classifier
at `~/.image_dimension_classifier.pickle` (default) and test by
`{test-image-path}`.

Note that all image files which this program uses is converted png format
and cropped at 100x100.

## Todo

- [x] `-p, --predict` Option
    - [x] Use `~/.image_dimension_classifier.pickle`
- [ ] `-t, --train` Option
- [ ] `-f` or `--pickle-filename` Option
- [ ] flake8
- [ ] write setup.py
- [ ] publish by pypi
- [ ] use travis

## Option

* --predict

Predict images dimension.

Usage is

```
$ classify_image_dimension --predict {image-path}
```

* --train

Train by images and put a pickle file.

Usage is

```
$ classify_image_dimension --train {image-path}
```

* -f, --pickle-filename

Indicate pickle file used by this program.
When this program uses the classifier, this pickle file is used
and saves it, this is used.

Usage is

```
$ classify_image_dimension ... --pickle-filename {pickle-filename}
```

* --not-convert

Not convert to small png file.

Usasge is

```
$ classify_image_dimension --not-convert ...
```

* --debug

Show all detailed error which this program is occured.

When you use this option, python error is outputted.
If you didn't use python, this might not be intuitive.
