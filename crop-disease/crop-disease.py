"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import geojson
from geojson.feature import Feature
import skimage.draw
import skimage
import skimage.color
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CropDiseaseConfig(Config):
    """Configuration for training on the crop disease dataset.
    Derives from the base Config class and overrides some values.
    """
    BACKBONE = "resnet50"
    
    # Give the configuration a recognizable name
    NAME = "crop-disease"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # TODO: Dynamic number of classes
    NUM_CLASSES = 1 + 1  

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1
    
    # test
    IMAGE_RESIZE_MODE = "none"
    
    # only one value because we use grayscale images
    # see https://github.com/matterport/Mask_RCNN/wiki#training-with-rgb-d-or-grayscale-images
    MEAN_PIXEL = np.array([0])

    


############################################################
#  Dataset
############################################################

class CropDiseaseDataset(utils.Dataset):

    def load_crop_disease(self, dataset_dir, subset):
        """Load feature subset of the crop disease dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        classes = json.load(open(os.path.join(dataset_dir, "diseases.json")))
        for key, value in classes.items():
            self.add_class("crop-disease", key, value)
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations = geojson.load(open(os.path.join(dataset_dir, "regions.json")))

        # Add images
        for feature in annotations.features:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. 
            polygons = feature.geometry.coordinates

            # load_mask() needs the image size to convert polygons to masks.
            for product in feature.properties["sentinelproducts"]:
                image_path = os.path.join(dataset_dir, feature.properties["id"] + "_" + product + ".tif")
                image = skimage.io.imread(image_path, plugin="pil")
                height, width = image.shape[:2]
    
                self.add_image(
                    "crop-disease",
                    image_id=feature.properties["id"] + "_" + product,
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons,
                    annotations=feature.properties["disease"])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a crop-disease dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "crop-disease":
            return super(self.__class__, self).load_mask(image_id)

        # read mask that was saved during generating the ndvi
        mask = np.load(image_info["path"].replace("tif", "mask")).mask != True
        
        print(mask.shape[-1])
        # Return mask, and array of class IDs of each instance. 
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32) # np.array(image_info["annotations"])

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "crop-disease":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
    def load_image(self, image_id):
        """Override to return grayscale image"""
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        return image


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CropDiseaseDataset()
    dataset_train.load_crop_disease(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CropDiseaseDataset()
    dataset_val.load_crop_disease(args.dataset, "val")
    dataset_val.prepare()

    print("Train everything")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='all')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to detect diseases')
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
        config = CropDiseaseConfig()
    else:
        class InferenceConfig(CropDiseaseConfig):
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
        # need to ignore first layer for grayscale images ub case of pre-trained weights
        model.load_weights(weights_path, by_name=True, exclude=["conv1",
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
        
    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        # load image
        image = skimage.io.imread("../sentineldata/products/ndvi/ef21dc70-7a1e-4bb5-99d0-6b7778e71377_S2A_MSIL2A_20180708T101031_N0208_R022_T32TPQ_20180708T133033.SAFE.tif")
        
        # Run detection
        results = model.detect([image], verbose=1)
        
        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
