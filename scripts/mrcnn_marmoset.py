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

import du, IPython as ip

# # Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
#
# # Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
# from mrcnn.config import Config
# from mrcnn import model as modellib, utils
#
# # Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#
# # Directory to save logs and model checkpoints, if not provided
# # through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


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

    # will call self.add_image("marmoset", image_id, image_path)
    #   each image will have an image, mask, resolution
    # masks will be stored as boolean H x W x nInstance
    # will overload load_mask(image_id)
    #   loads mask directly; also rescales if H, W different from stated
    #
    # how to represent no target?

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


# def color_splash(image, mask):
#   """Apply color splash effect.
#   image: RGB image [height, width, 3]
#   mask: instance segmentation mask [height, width, instance count]
#
#   Returns result image.
#   """
#   # Make a grayscale copy of the image. The grayscale copy still
#   # has 3 RGB channels, though.
#   gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
#   # Copy color pixels from the original color image where mask is set
#   if mask.shape[-1] > 0:
#     # We're treating all instances as one, so collapse the mask into one layer
#     mask = (np.sum(mask, -1, keepdims=True) >= 1)
#     splash = np.where(mask, image, gray).astype(np.uint8)
#   else:
#     splash = gray.astype(np.uint8)
#   return splash

def detect(model, image_path, outpath):
  cols = du.diffcolors(100, alpha=0.5)

  # Run model detection and generate the color splash effect
  print("Running on {}".format(args.image))

  img = du.imread(image_path)

  # Detect objects
  r = model.detect([img], verbose=1)[0]

  # draw masks
  masks = r['masks']
  nMasks = masks.shape[-1]
  for nM in range(nMasks):
    img = du.DrawOnImage(img, np.nonzero(masks[:,:,nM]), cols[nM])

  # save image
  base = du.fileparts(image_path)[1]
  du.imwrite(img, f'{outpath}/{base}.jpg')

  # Read images
  # imgs = du.GetImgPaths(image_path)
  # for imgPath in imgs:
  #   img = du.imread(imgPath)
  #
  #   # Detect objects
  #   r = model.detect([img], verbose=1)[0]
  #
  #   # draw masks
  #   masks = r['masks']
  #   nMasks = masks.shape[-1]
  #   for nM in range(nMasks):
  #     img = du.DrawOnImage(img, np.nonzero(masks[:,:,nM]), cols[nM])
  #   
  #   # save image
  #   base = du.fileparts(imgPath)[1]
  #   du.imwrite(img, f'{outpath}/{base}.jpg')


############################################################
#  Training
############################################################

if __name__ == '__main__':
  import argparse

  # Parse command line arguments
  parser = argparse.ArgumentParser(
    description='Train Mask R-CNN to detect marmosets.')
  parser.add_argument("command",
            metavar="<command>",
            help="'train' or 'splash'")
  parser.add_argument('--dataset', required=False,
            metavar="/path/to/marmoset/dataset/",
            help='Directory of the Marmoset dataset')
  parser.add_argument('--weights', required=True,
            metavar="/path/to/weights.h5",
            help="Path to weights .h5 file or 'coco'")
  parser.add_argument('--logs', required=False,
            default=DEFAULT_LOGS_DIR,
            metavar="/path/to/logs/",
            help='Logs and checkpoints directory (default=logs/)')
  parser.add_argument('--image', required=False,
            metavar="path or URL to image",
            help='Image to apply the color splash effect on')
  parser.add_argument('--output', default='output',
            help='Path to output detections')
  args = parser.parse_args()

  # Validate arguments
  if args.command == "train":
    assert args.dataset, "Argument --dataset is required for training"
  elif args.command == "splash":
    assert args.image or args.video,\
         "Provide --image or --video to apply color splash"

  print("Weights: ", args.weights)
  print("Dataset: ", args.dataset)
  print("Logs: ", args.logs)

  # Configurations
  if args.command == "train":
    config = MarmosetConfig()
  else:
    class InferenceConfig(MarmosetConfig):
      # Set batch size to 1 since we'll be running inference on
      # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1
    config = InferenceConfig()
  config.display()

  # Create model
  if args.command == "train":
    model = modellib.MaskRCNN(mode="training", config=config,
                  model_dir=args.logs)
  else:
    model = modellib.MaskRCNN(mode="inference", config=config,
                  model_dir=args.logs)

  # Select weights file to load
  if args.weights.lower() == "coco":
    weights_path = COCO_WEIGHTS_PATH
    # Download weights file
    if not os.path.exists(weights_path):
      utils.download_trained_weights(weights_path)
  elif args.weights.lower() == "last":
    # Find last trained weights
    weights_path = model.find_last()
  elif args.weights.lower() == "imagenet":
    # Start from ImageNet trained weights
    weights_path = model.get_imagenet_weights()
  else:
    weights_path = args.weights

  # Load weights
  print("Loading weights ", weights_path)
  if args.weights.lower() == "coco":
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
      "mrcnn_class_logits", "mrcnn_bbox_fc",
      "mrcnn_bbox", "mrcnn_mask"])
  else:
    model.load_weights(weights_path, by_name=True)

  # Train or evaluate
  if args.command == "train":
    train(model)
  elif args.command == "detect":
    detect(model, args.image, args.output)
  else:
    print("'{}' is not recognized. "
        "Use 'train' or 'detect'".format(args.command))
