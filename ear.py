"""
Mask R-CNN
Train on the toy Ear dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 ear.py train --dataset=/path/to/ear/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 ear.py train --dataset=/path/to/ear/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 ear.py train --dataset=/path/to/ear/dataset --weights=imagenet

    # Apply color splash to an image
    python3 ear.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 ear.py splash --weights=last --video=<URL or path to file>
"""

import datetime
import os
import sys
from glob import glob

import cv2
import numpy as np
import skimage.draw

from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class EarConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ear"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + ear

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

############################################################
#  Dataset
############################################################


class EarDataset(utils.Dataset):

    def load_ear(self, awe_dir, folder):
        """Load a subset of the Ear dataset.
        awe_dir: Root directory of the dataset.
        folder: Subset to load: train or test
        """
        # Add classes. We have only one class to add.
        self.add_class("ear", 1, "ear")

        # Train or validation dataset?
        assert folder in ["train", "test"]
        src_dir = os.path.join(awe_dir, folder)
        annot_dir = os.path.join(awe_dir, '{}annot'.format(folder))
        bb_dir = os.path.join(awe_dir, '{}annot_rect'.format(folder))
        mask_dir = os.path.join(awe_dir, '{}_masks'.format(folder))
        images = glob('{}/**/*'.format(src_dir), recursive=True)
        images = [i for i in images if os.path.exists(i.replace(src_dir, annot_dir)) and os.path.exists(i.replace(src_dir, bb_dir))]

        for path in images:
            img = cv2.imread(path)
            mask = cv2.imread(path.replace(src_dir, annot_dir))
            bb = cv2.imread(path.replace(src_dir, bb_dir))
            w, h, c = img.shape

            gray = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            mask_path = path.replace(src_dir, mask_dir) + '.npy'

            if not os.path.exists(mask_path):
                if not os.path.exists(os.path.dirname(mask_path)):
                    os.makedirs(os.path.dirname(mask_path))

                masks = []
                for c in contours:
                    m = np.full((w, h), False)
                    x1 = min([p[0][0] for p in c if p[0][0] >= 0])
                    x2 = max([p[0][0] for p in c if p[0][0] >= 0])
                    y1 = min([p[0][1] for p in c if p[0][1] >= 0])
                    y2 = max([p[0][1] for p in c if p[0][1] >= 0])
                    m[x1:x2, y1:y2] = mask[x1:x2, y1:y2, 1] > 0
                    masks.append(m)

                mask = np.stack(masks, axis=2)
                with open(mask_path, 'wb+') as file:
                    np.save(file, mask)

            self.add_image(
                "ear",
                image_id=path.replace(src_dir, '')[1:],
                path=path,
                mask=mask_path,
                objects=len(contours),
                width=w, height=h
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a ear dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ear":
            return super(self.__class__, self).load_mask(image_id)

        with open(image_info['mask'], 'rb') as file:
            mask = np.load(file)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.uint8), np.ones([image_info['objects']], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ear":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = EarDataset()
    dataset_train.load_ear(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = EarDataset()
    dataset_val.load_ear(args.dataset, "test")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


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
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(
            datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2

        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(
            datetime.datetime.now())
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
        description='Train Mask R-CNN to detect ears.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/ear/dataset/",
                        help='Directory of the Ear dataset')
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
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
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
        config = EarConfig()
    else:
        class InferenceConfig(EarConfig):
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
    elif args.weights.lower() == "random":
        pass
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
