# Mask R-CNN for Object Detection and Segmentation

This is [Mask R-CNN](https://arxiv.org/abs/1703.06870) implementation is based on the work of [Matterport](https://github.com/matterport/Mask_RCNN) which was developed on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

This project was developed for my [master thesis](https://github.com/simhue/Mask_RCNN/blob/master/texsrc/main.pdf). It evaluated abitility Mask R-CNN to recognize crop diseases via Sentinel-2 MSI products.

The work is splitted in two parts. The first part [sentineldata.py](https://github.com/simhue/Mask_RCNN/blob/master/sentineldata/sentineldata.py) downloads Sentinel products, calculates the NDVI and crops out the RoI. Also it creates a binary mask representing the RoI for later training. It's stored separately since it doesn't need to be recreated before the images are loaded into the model. For this to work, [annotated RoIs](https://github.com/simhue/Mask_RCNN/wiki/Annotation) as GeoJsons are necessary. 

The second part is the actual model in [crop-disease.py](https://github.com/simhue/Mask_RCNN/blob/master/crop-disease/crop-disease.py). It loads the dataset and feeds the model with it. After the succesful training, the model can be used for detection tasks. For further information on the model implementation details, please consider reading the documentation and Jupyter Notebooks of the original Github.

## Installation

1. Clone this repository
2. Install dependecies `pip3 install -r requirements.txt`
3. Create RoI and store it into `./datasets/roi.geojson`
3. Create [Copernicus Open Access Hub](https://scihub.copernicus.eu/) account
4. Navigate to `./sentineldata/` and run `python3 sentineldata.py username password` using the account credentials
5. The script will download und process the sentinel products, if any are available
6. For training, navigate to `./crop-disease/` and run `python3 crop-disease.py --weights=/path/to/weights.h5|coco|imagenet --dataset=../datasets train` - coco and imagenet will download pre-trained weights
7. For detection `python3 crop-disease.py --weights=/path/to/weights.h5|coco|imagenet --image=path/to/ndvi/image.tif detect` -- WIP -- It's still in development and paths were hard coded.
  

