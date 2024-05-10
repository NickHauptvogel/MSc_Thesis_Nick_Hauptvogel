import argparse

import keras
from keras.datasets import cifar10, cifar100
import numpy as np
import csv
import io
import os

from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds


def load_cifar(num_classes, subtract_pixel_mean, debug):
    """
    Load CIFAR-10 or CIFAR-100 dataset.
    :param num_classes: Number of classes in dataset
    :param subtract_pixel_mean: Subtract pixel mean from images
    :param debug: Use a small subset of the dataset
    """
    if num_classes == 10:
        dataloader = cifar10
    elif num_classes == 100:
        dataloader = cifar100
    else:
        raise ValueError('Unknown number of classes')
    (x_train, y_train), (x_test, y_test) = dataloader.load_data()

    if debug:
        x_train = x_train[:1000]
        y_train = y_train[:1000]
        x_test = x_test[:100]
        y_test = y_test[:100]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


# The following code is taken and adapted from
# https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/datasets/diabetic_retinopathy_dataset_utils.py
#
# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

_CITATION = """\
@ONLINE {kaggle-diabetic-retinopathy,
    author = "Kaggle and EyePacs",
    title  = "Kaggle Diabetic Retinopathy Detection",
    month  = "jul",
    year   = "2015",
    url    = "https://www.kaggle.com/c/diabetic-retinopathy-detection/data"
}
"""
_URL_TEST_LABELS = (
    "https://storage.googleapis.com/kaggle-forum-message-attachments/"
    "90528/2877/retinopathy_solution.csv")
_BLUR_BTGRAHAM_DESCRIPTION_PATTERN = (
    "A variant of the processing method used by the winner of the 2015 Kaggle "
    "competition: images are resized so that the radius of an eyeball is "
    "{} pixels, then receive a Gaussian blur-based normalization with Kernel "
    "standard deviation along the X-axis of {}. Then they are cropped to 90% "
    "of the radius, and finally they are encoded with 72 JPEG quality.")


class UBDiabeticRetinopathyDetectionConfig(tfds.core.BuilderConfig):
    """BuilderConfig for DiabeticRetinopathyDetection."""

    def __init__(self, target_pixels=None, blur_constant=None, **kwargs):
        """BuilderConfig for DiabeticRetinopathyDetection.

    Args:
      target_pixels: If given, rescale the images so that the total number of
        pixels is roughly this value.
      blur_constant: Constant used to vary the Kernel standard deviation in
        smoothing the image with Gaussian blur.
      **kwargs: keyword arguments forward to super.
    """
        super(UBDiabeticRetinopathyDetectionConfig, self).__init__(
            version=tfds.core.Version("3.0.0"),
            **kwargs)
        self._target_pixels = target_pixels
        self._blur_constant = blur_constant

    @property
    def target_pixels(self):
        return self._target_pixels

    @property
    def blur_constant(self):
        return self._blur_constant


class UBDiabeticRetinopathyDetection(tfds.core.GeneratorBasedBuilder):
    """Diabetic retinopathy detection."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You have to download this dataset from Kaggle.
  https://www.kaggle.com/c/diabetic-retinopathy-detection/data
  After downloading, unpack the test.zip file into test/ directory in manual_dir
  and sample.zip to sample/. Also unpack the sampleSubmissions.csv and
  trainLabels.csv.
  """

    BUILDER_CONFIGS = [
        UBDiabeticRetinopathyDetectionConfig(
            name="blur-30-btgraham-300",
            description=_BLUR_BTGRAHAM_DESCRIPTION_PATTERN.format(300, 300 // 30),
            blur_constant=30,
            target_pixels=300)
    ]

    def __init__(self):
        super(UBDiabeticRetinopathyDetection, self).__init__(
            config=UBDiabeticRetinopathyDetection.BUILDER_CONFIGS[0]
        )

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="A large set of high-resolution retina images taken under "
                        "a variety of imaging conditions.",
            features=tfds.features.FeaturesDict({
                "name": tfds.features.Text(),  # patient ID + eye. eg: "4_left".
                "image": tfds.features.Image(),
                # From 0 (no DR - saine) to 4 (Proliferative DR). -1 means no label.
                "label": tfds.features.ClassLabel(num_classes=5),
            }),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.manual_dir
        test_labels_path = dl_manager.download(_URL_TEST_LABELS)
        if tf.io.gfile.isdir(test_labels_path):
            # While testing: download() returns the dir containing the tests files.
            test_labels_path = os.path.join(test_labels_path,
                                            "retinopathy_solution.csv")
        return [
            tfds.core.SplitGenerator(
                name="sample",  # 10 images, to do quicktests using dataset.
                gen_kwargs={
                    "images_dir_path": os.path.join(path, "sample"),
                },
            ),
            tfds.core.SplitGenerator(
                name="train",
                gen_kwargs={
                    "images_dir_path": os.path.join(path, "train"),
                    "csv_path": os.path.join(path, "trainLabels.csv"),
                    # CSV of the train split does not have the "Usage" column.
                    # 35,126 examples.
                    "csv_usage": None,
                },
            ),
            tfds.core.SplitGenerator(
                name="validation",
                gen_kwargs={
                    "images_dir_path": os.path.join(path, "test"),
                    "csv_path": test_labels_path,
                    # Validation split corresponds to the public leaderboard data.
                    # 10,906 examples.
                    "csv_usage": "Public",
                },
            ),
            tfds.core.SplitGenerator(
                name="test",
                gen_kwargs={
                    "images_dir_path": os.path.join(path, "test"),
                    "csv_path": test_labels_path,
                    # Test split corresponds to the public leaderboard data.
                    # 42,670 examples.
                    "csv_usage": "Private",
                },
            ),
        ]

    def generate_examples(self, images_dir_path, csv_path=None, csv_usage=None, start=None, stop=None):
        return self._generate_examples(images_dir_path, csv_path, csv_usage, start, stop)

    def _generate_examples(self, images_dir_path, csv_path=None, csv_usage=None, start=None, stop=None):
        """Yields Example instances from given CSV.

    Args:
      images_dir_path: path to dir in which images are stored.
      csv_path: optional, path to csv file with two columns: name of image and
        label. If not provided, just scan image directory, don't set labels.
      csv_usage: optional, subset of examples from the csv file to use based on
        the "Usage" column from the csv.
    """
        if csv_path:
            with tf.io.gfile.GFile(csv_path) as csv_f:
                reader = csv.DictReader(csv_f)
                data = [(row["image"], int(row["level"]))
                        for row in reader
                        if csv_usage is None or row["Usage"] == csv_usage]
        else:
            data = [(fname[:-5], -1)
                    for fname in tf.io.gfile.listdir(images_dir_path)
                    if fname.endswith(".jpeg")]

        if start is not None and stop is not None:
            if start >= len(data):
                raise ValueError("Start index is out of range.")
            if stop > len(data):
                print("Setting stop index to the end of the dataset.")
                stop = len(data)
            data = data[start:stop]

        logging.info("Using BuilderConfig %s.", self.builder_config.name)

        for name, label in data:
            image_filepath = "%s/%s.jpeg" % (images_dir_path, name)
            record = {
                "name": name,
                "image": self._process_image(image_filepath),
                "label": label,
            }
            yield name, record

    def _process_image(self, filepath):
        with tf.io.gfile.GFile(filepath, mode="rb") as image_fobj:
            if self.builder_config.name.startswith("btgraham"):
                return _btgraham_processing(  # pylint: disable=protected-access
                    image_fobj=image_fobj,
                    filepath=filepath,
                    target_pixels=self.builder_config.target_pixels,
                    blur_constant=30,
                    crop_to_radius=True)
            elif self.builder_config.name.startswith("blur"):
                return _btgraham_processing(  # pylint: disable=protected-access
                    image_fobj=image_fobj,
                    filepath=filepath,
                    target_pixels=self.builder_config.target_pixels,
                    blur_constant=self.builder_config.blur_constant,
                    crop_to_radius=True)
            else:
                return _resize_image_if_necessary(  # pylint: disable=protected-access
                    image_fobj=image_fobj,
                    target_pixels=self.builder_config.target_pixels)


def _resize_image_if_necessary(image_fobj, target_pixels=None):
    """Resize an image to have (roughly) the given number of target pixels.

  Args:
    image_fobj: File object containing the original image.
    target_pixels: If given, number of pixels that the image must have.
  Returns:
    A file object.
  """
    if target_pixels is None:
        return image_fobj

    cv2 = tfds.core.lazy_imports.cv2
    # Decode image using OpenCV2.
    image = cv2.imdecode(
        np.frombuffer(image_fobj.read(), dtype=np.uint8), flags=3)
    # Get image height and width.
    height, width, _ = image.shape
    actual_pixels = height * width
    if actual_pixels > target_pixels:
        factor = np.sqrt(target_pixels / actual_pixels)
        image = cv2.resize(image, dsize=None, fx=factor, fy=factor)
    # Encode the image with quality=72 and store it in a BytesIO object.
    _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return io.BytesIO(buff.tobytes())


def _btgraham_processing(image_fobj,
                         filepath,
                         target_pixels,
                         blur_constant,
                         crop_to_radius=False):
    """Process an image as the winner of the 2015 Kaggle competition.

  Args:
    image_fobj: File object containing the original image.
    filepath: Filepath of the image, for logging purposes only.
    target_pixels: The number of target pixels for the radius of the image.
    blur_constant: Constant used to vary the Kernel standard deviation in
      smoothing the image with Gaussian blur.
    crop_to_radius: If True, crop the borders of the image to remove gray areas.

  Returns:
    A file object.
  """
    cv2 = tfds.core.lazy_imports.cv2
    # Decode image using OpenCV2.
    image = cv2.imdecode(
        np.frombuffer(image_fobj.read(), dtype=np.uint8), flags=3)
    # Process the image.
    image = _scale_radius_size(image, filepath, target_radius_size=target_pixels)
    image = _subtract_local_average(
        image, target_radius_size=target_pixels, blur_constant=blur_constant)
    image = _mask_and_crop_to_radius(
        image,
        target_radius_size=target_pixels,
        radius_mask_ratio=0.9,
        crop_to_radius=crop_to_radius)
    # Consistently resize the image to 512x512.
    image = cv2.resize(image, dsize=(512, 512))
    # Encode the image with quality=72 and store it in a BytesIO object.
    _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return io.BytesIO(buff.tobytes())


def _scale_radius_size(image, filepath, target_radius_size):
    """Scale the input image so that the radius of the eyeball is the given."""
    cv2 = tfds.core.lazy_imports.cv2
    x = image[image.shape[0] // 2, :, :].sum(axis=1)
    r = (x > x.mean() / 10.0).sum() / 2.0
    if r < 1.0:
        # Some images in the dataset are corrupted, causing the radius heuristic to
        # fail. In these cases, just assume that the radius is the height of the
        # original image.
        logging.info("Radius of image \"%s\" could not be determined.", filepath)
        r = image.shape[0] / 2.0
    s = target_radius_size / r
    return cv2.resize(image, dsize=None, fx=s, fy=s)


def _subtract_local_average(image, target_radius_size, blur_constant=30):
    cv2 = tfds.core.lazy_imports.cv2
    image_blurred = cv2.GaussianBlur(image, (0, 0),
                                     target_radius_size / blur_constant)
    image = cv2.addWeighted(image, 4, image_blurred, -4, 128)
    return image


def _mask_and_crop_to_radius(image,
                             target_radius_size,
                             radius_mask_ratio=0.9,
                             crop_to_radius=False):
    """Mask and crop image to the given radius ratio."""
    cv2 = tfds.core.lazy_imports.cv2
    mask = np.zeros(image.shape)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    radius = int(target_radius_size * radius_mask_ratio)
    cv2.circle(mask, center=center, radius=radius, color=(1, 1, 1), thickness=-1)
    image = image * mask + (1 - mask) * 128
    if crop_to_radius:
        x_max = min(image.shape[1] // 2 + radius, image.shape[1])
        x_min = max(image.shape[1] // 2 - radius, 0)
        y_max = min(image.shape[0] // 2 + radius, image.shape[0])
        y_min = max(image.shape[0] // 2 - radius, 0)
        image = image[y_min:y_max, x_min:x_max, :]
    return image


def preprocess_retinopathy(output_folder, folder, csv_name, dataset_path):

    dataset = UBDiabeticRetinopathyDetection()
    os.makedirs(f"{dataset_path}/{output_folder}", exist_ok=True)
    for i, example in enumerate(dataset.generate_examples(f"{dataset_path}/{folder}", f"{dataset_path}/{csv_name}")):
        print(example[0], "({})".format(i))
        # Store the image in a file
        with open(f"{dataset_path}/{output_folder}/{example[0]}.jpeg", "wb") as f:
            f.write(example[1]['image'].read())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess the Diabetic Retinopathy dataset')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--train', action='store_true', help='Preprocess the training set')
    parser.add_argument('--test', action='store_true', help='Preprocess the test set')
    args = parser.parse_args()

    if args.train:
        preprocess_retinopathy("train", "train", "trainLabels.csv", args.dataset_path)
    if args.test:
        preprocess_retinopathy("test", "test", "retinopathy_solution.csv", args.dataset_path)