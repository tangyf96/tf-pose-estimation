import argparse
import sys
import time
import os
import glob
from tf_pose import common
import numpy as np
#from tf_pose.estimator import TfPoseEstimator
from tf_pose.new_estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# write the body points of each human into a file
def write_to_file(image, human_info, im_file):
    op_imfile = im_file.replace(im_file[-4:], ".txt")
    f = open(op_imfile, "w")
    f.write(str(len(human_info)))
    f.write("\n\n")
    
    for human in human_info:
        for (information,location) in human:
            f.write(information+": "+str(location))
            f.write("\n")
        f.write("\n")
    f.close()
    
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image_dir', type=str, default='images/xahid_youya/input/selected')
    parser.add_argument('--image_type', type=str, default='*.jpg')
    
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
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
    
    count=0
    persons=[]
    color=[]
    
    TEST_IMAGE_PATHS = glob.glob(os.path.join(args.image_dir, args.image_type))
    TEST_IMAGE_PATHS.sort(key=lambda f: int(filter(str.isdigit, f)))
    t = time.time()
    for im_file in TEST_IMAGE_PATHS:
        print
        print('Processing = {0}'.format(im_file))        

        img = common.read_imgfile(im_file, None, None)
        if img is None:
            print('Image can not be read, path= {0}'.format(im_file))
            continue
        
        humans = e.inference(img, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        image, human_info = TfPoseEstimator.draw_humans(im_file, img, humans, imgcopy=False)
        write_to_file(image, human_info, im_file)
        
    elapsed = time.time() - t
    print
    print("total Time taken for processing {0} images: {0} seconds".format(len(TEST_IMAGE_PATHS), elapsed))
