"""
Mask R-CNN
Train on the toy Marmoset dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
     the command line as such:

  # Train a new model starting from pre-trained COCO weights
  python3 marmoset.py train --dataset=/path/to/marmoset/dataset --weights=coco

  # Resume training a model that you had trained earlier
  python3 marmoset.py train --dataset=/path/to/marmoset/dataset --weights=last

  # Train a new model starting from ImageNet weights
  python3 marmoset.py train --dataset=/path/to/marmoset/dataset --weights=imagenet

  # Apply color splash to an image
  python3 marmoset.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

  # Apply color splash to video using the last weights you trained
  python3 marmoset.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import du, ffmpegu as fu
import cv2
from pycocotools import mask as pcMask

class MarmosetConfig(Config):
  """Configuration for training on the toy  dataset.
  Derives from the base Config class and overrides some values.
  """
  # Give the configuration a recognizable name
  NAME = "marmoset"

  # We use a GPU with 12GB memory, which can fit two images.
  # Adjust down if you use a smaller GPU.
  IMAGES_PER_GPU = 2

  # Number of classes (including background)
  NUM_CLASSES = 1 + 1  # Background + marmoset

  # Number of training steps per epoch
  STEPS_PER_EPOCH = 100

  # Skip detections with < 90% confidence
  DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################
class MarmosetDataset(utils.Dataset):
  def load_marmoset(self, dataset_dir, subset):
    """Load a subset of the Marmoset dataset.
    dataset_dir: Root directory of the dataset.
    subset: Subset to load: train or val
    """
    # Add classes. We have only one class to add.
    self.add_class("marmoset", 1, "marmoset")

    # Train or validation dataset?
    assert subset in ["train", "val"]
    dataset_dir = os.path.join(dataset_dir, subset)

    imgs = du.GetImgPaths(f'{dataset_dir}/rgb')
    masks = du.GetFilePaths(f'{dataset_dir}/mask', 'gz')
    assert len(imgs) == len(masks)

    for idx, path in enumerate(imgs):
      base = du.fileparts(path)[1]
      self.add_image('marmoset', idx, path, base=base)

  def load_mask(self, image_id):
    """Generate instance masks for an image.
     Returns:
    masks: A bool array of shape [height, width, instance count] with
      one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    info = self.image_info[image_id]
    if info["source"] != "marmoset":
      return super(self.__class__, self).load_mask(image_id)

    imPath = info['path']
    imPathBase = du.fileparts(imPath)[0]
    imBase = info['base']
    maskPath = f'{imPathBase}/../mask/{imBase}.gz'
    mask = du.load(maskPath)
    class_ids = np.ones([mask.shape[-1]], dtype=np.int32)
    return mask, class_ids

  def image_reference(self, image_id):
    """Return the path of the image."""
    info = self.image_info[image_id]
    if info["source"] == "marmoset":
      return info["path"]
    else:
      super(self.__class__, self).image_reference(image_id)

def train(model):
  """Train the model."""
  # Training dataset.
  dataset_train = MarmosetDataset()
  dataset_train.load_marmoset(args.dataset, "train")
  dataset_train.prepare()

  # Validation dataset
  dataset_val = MarmosetDataset()
  dataset_val.load_marmoset(args.dataset, "val")
  dataset_val.prepare()

  # *** This training schedule is an example. Update to your needs ***
  # Since we're using a very small dataset, and starting from
  # COCO trained weights, we don't need to train too long. Also,
  # no need to train all layers, just the heads should do it.
  # print("Training network heads")
  # model.train(dataset_train, dataset_val,
  #       learning_rate=config.LEARNING_RATE,
  #       epochs=30,
  #       layers='heads')
  print("Training full network")
  model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=1000,
        layers='all')

def _saveDetection(detectionResult, outPath, **kwargs):
  draw = kwargs.get('draw', False)
  singleColor = kwargs.get('singleColor', False)

  if draw:
    cols = du.diffcolors(100, alpha=0.5)

    masks = detectionResult['masks']
    nMasks = masks.shape[-1]
    if singleColor:
      img = du.DrawOnImage(img, np.nonzero(np.sum(masks, axis=2)), cols[0])
    else:
      for nM in range(nMasks):
        img = du.DrawOnImage(img, np.nonzero(masks[:,:,nM]), cols[nM])

    # save image
    du.imwrite(img, outPath)

  else:
    enc = pcMask.encode(np.asfortranarray(
      detectionResult['masks']).astype(np.uint8))
    du.save(outPath, {'masks': enc, 'scores': detectionResult['scores']})

def detect(weightPath, imgs, outs, **kwargs):
  draw = kwargs.get('draw', False)
  ss = kwargs.get('ss', None)
  minConf = kwargs.get('minConf', 0.7)

  class InferenceConfig(MarmosetConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_NMS_THRESHOLD = 0.9
    DETECTION_MIN_CONFIDENCE = minConf

  model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(),
    model_dir='.')
  model.load_weights(weightPath, by_name=True)

  if type(imgs) == str and type(outs) == str:
    imgs = [imgs, ]
    outs = [outs, ]

  for imgPath, outPath in zip(imgs, outs):
    print(imgPath, outPath)
    ext = du.fileparts(imgPath)[2].lower()
    if ext == '.mp4' or ext == '.avi' or ext == '.mov':
      if ss is not None:
        img = fu.GetRGBFrameByTimeNumpy(imgPath, ss)
        r = model.detect([img], verbose=1)[0]
        _saveDetection(r, outPath, **kwargs)
      else:
        video = cv2.VideoCapture(imgPath)
        success, cnt = (True, 0)
        while success:
          du.tic()
          success, image = video.read()
          r = model.detect([image], verbose=1)[0]
          outName = f'{outPath}-{cnt:08}'
          if draw: outName += '.jpg'
          _saveDetection(r, outName, **kwargs)
          print(du.toc())
          cnt += 1
        continue
    else:
      img = du.imread(imgPath)
      r = model.detect([img], verbose=1)[0]
      _saveDetection(r, outPath, **kwargs)
