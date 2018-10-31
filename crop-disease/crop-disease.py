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
from skimage.external import tifffile
import skimage.color
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from IPython.display import display, Image
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

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # TODO: Dynamic number of classes
    NUM_CLASSES = 1 + 1  

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 400
    
    # IMAGE_CHANNEL_COUNT = 1

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    
    USE_MINI_MASK = False
    
    DETECTION_MIN_CONFIDENCE = 0.8
    
    MEAN_PIXEL = np.array([52, 52, 52])

############################################################
#  Dataset
############################################################


class CropDiseaseDataset(utils.Dataset):
    def load_crop_disease(self, dataset_dir, subset):
        """Load feature subset of the crop disease dataset.
        subset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        subset_dir = os.path.join(dataset_dir, subset)
        
        # Add classes. We have only one class to add.
        self.add_class("crop-disease", 1, "infection")
        
        # Load annotations
        annotations = geojson.load(open(os.path.join(subset_dir, "regions.geojson")))

        # Add images
        for feature in annotations.features:
            # load_mask() needs the image size to convert polygons to masks.
            image_path = os.path.join(subset_dir, feature.properties["id"] + ".tif")

            self.add_image(
                "crop-disease",
                image_id=feature.properties["id"],
                path=image_path)


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

        mask = self.get_mask(image_info)

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    def get_mask(self, image_info):
        # read mask that was saved during generating the ndvi
        masked_arr = np.load(image_info["path"].replace("tif", "mask"))
        # the mask needs to be inverted and reshaped since it comes in a shape like [1, height, width]
        # but we need [height, width, 1] (the one represents the number of instances,
        # but since we only have one instance of an image we use 1)
        mask = masked_arr != True
        return np.expand_dims(mask[0], axis=2)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "crop-disease":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

            
    def load_image(self, image_id):
        """Override to return image"""
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path']) 
        rgb_image = skimage.color.gray2rgb(image * 255)
        return rgb_image 


class MultipleDiseasesDataset(CropDiseaseDataset):
    def load_crop_disease(self, dataset_dir, subset):
        """Load feature subset of the crop disease dataset.
        subset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        subset_dir = os.path.join(dataset_dir, subset)
        
        # Add classes. We have only one class to add.
        classes = json.load(open(os.path.join(dataset_dir, "diseases.json")))
        for key, value in classes.items():
            self.add_class("crop-disease", key, value)
        
        # Load annotations
        annotations = geojson.load(open(os.path.join(subset_dir, "regions.geojson")))

        # Add images
        for feature in annotations.features:
            # load_mask() needs the image size to convert polygons to masks.
            image_path = os.path.join(subset_dir, feature.properties["id"] + ".tif")

            self.add_image(
                "crop-disease",
                image_id=feature.properties["id"],
                path=image_path,
                annotation=feature.properties["disease"])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "crop-disease":
            return super(self.__class__, self).load_mask(image_id)

        mask = self.get_mask(image_info)

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.array([image_info["annotation"]]).astype(np.int32)


def train(model):
    """Train the model."""
    # Training dataset.
    print("Loading training dataset")
    dataset_train = CropDiseaseDataset()
    dataset_train.load_crop_disease(args.dataset, "train")
    dataset_train.prepare()
    
    print("Image Count: {}".format(len(dataset_train.image_ids)))
    print("Class Count: {}".format(dataset_train.num_classes))
    for i, info in enumerate(dataset_train.class_info):
        print("{:3}. {:50}".format(i, info['name']))
        
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Validation dataset
    print("Loading validation set")
    dataset_val = CropDiseaseDataset()
    dataset_val.load_crop_disease(args.dataset, "val")
    dataset_val.prepare()

    print("Image Count: {}".format(len(dataset_val.image_ids)))
    print("Class Count: {}".format(dataset_val.num_classes))
    for i, info in enumerate(dataset_val.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    print("Start training")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='all')
    
    
    
    
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.color.gray2rgb(skimage.io.imread(image_path))
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
        print("Saved to ", file_name)

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
            #IMAGE_RESIZE_MODE = "square"
            #IMAGE_MIN_DIM = 2048
            

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
        # need to ignore first layer for grayscale images in case of pre-trained weights
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
        
    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        # load image
        # detect_and_color_splash(model, image_path="ndvi.tif")
        # Run detection
         
        image = skimage.color.gray2rgb(skimage.io.imread("ndvi-2.tif") * 255)
        results = model.detect([image], verbose=1)
        
        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                       ["BG", "Anthracnose"], r['scores'])
        
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
