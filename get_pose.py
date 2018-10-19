import argparse
import logging
import sys
import time
import os
import glob
from tf_pose import common
import cv2
import numpy as np
from tf_pose.new_estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    PATH_TO_TEST_IMAGES_DIR = 'images/xahid_youya/input'
    TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))
    for im_file in TEST_IMAGE_PATHS:
        print(im_file)
        parser = argparse.ArgumentParser(description='tf-pose-estimation run')
        parser.add_argument('--image', type=str, default=im_file)
        parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

        parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
        args = parser.parse_args()
        w, h = model_wh(args.resize)
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

        # estimate human poses from a single image !
        image = common.read_imgfile(args.image, None, None)
        if image is None:
            logger.error('Image can not be read, path=%s' % args.image)
            sys.exit(-1)
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

        image = TfPoseEstimator.draw_humans(im_file, image, humans, imgcopy=False)
        im_file1=im_file.replace('images/xahid_youya/input','')
        print(im_file1)
        op_imfile = os.path.join('images/xahid_youya/output', im_file1)
        print(op_imfile)
        cv2.imwrite(op_imfile, image)
