import argparse
import logging
import sys
import time
import os
import glob
from tf_pose import common
import cv2
import numpy as np
import math
from tf_pose.new_estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import random
import numpy
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
#write the body points of each human into a file
def write_to_file(persons,op_imfile,count):
    op_imfile = 'images/xahid_youya/output/'+(op_imfile.replace("images/xahid_youya/test_openpose","")).replace(".png",".txt")
    f = open(op_imfile, "w")
    f.write(str(len(human_info)))
    f.write("\n\n")
    for i in persons:
        if i.get("last_update")==count:
            for (information,location) in i.get("body_points"):
                if location!=None:
                    x,y=location
                    f.write(information+": "+str(x)+", "+str(y))
                else:
                    f.write(information+": "+"None")
                f.write("\n")
            f.write("\n\n")
    f.close()
#put the body points and features of each human into a dictionary below
def feature_extraction(human_info,human_boxes,feature,count):
    persons=[]
    count=0
    for i in  range(len(human_info)):
        some_human={"id":count,"body_points":human_info[i],"last_update":count,"bound_box":human_boxes[i],"feature":feature[i]}
        persons.append(some_human)
    return persons
#check the body points distance between two human beings
def square_error_checker(first_human,second_human):
    error=0
    first_location=[]
    second_location=[]
    for i in range(len(first_human)):
        (information1,location1)=first_human[i]
        (information2,location2)=second_human[i]
        #print(location2)
        if (location1!=None and location2!=None):
            (x1,y1)=location1
            (x2,y2)=location2
            first_location.append(location1)
            second_location.append(location2)
    return mean_squared_error(first_location, second_location)
#draw the body points of each person in each frame
def draw_persons(npimg,count,persons,color):
    while len(color)<len(persons):
        color.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    for i in range(len(persons)):
        if persons[i].get("last_update")==count:
            for (information,location) in persons[i].get("body_points"):
                if location!=None:
                    cv2.circle(npimg, location, 3,color[i], thickness=3, lineType=8, shift=0)
    return (npimg,color)
#compare the distance of body points of each person in the previous frame with the target person in the next frame and then find the feature error between the person we find in the previous frame with the target person in the next frame
def find_min(first_human,persons,count,exclude):
    #print(first_human.get("body_points"))
    min_error=square_error_checker(persons[0].get("body_points"),first_human.get("body_points"))
    #min_error=100000
    target=0
    for i in range(len(persons)):
        #print(persons[i].get("body_points"))
        if i not in exclude:
            current_error=square_error_checker(first_human.get("body_points"),persons[i].get("body_points"))
            if current_error<=min_error:
                min_error=current_error
                target=i
    target_feature=persons[target].get("feature")
    error=distance.euclidean(target_feature, first_human.get("feature"))
    #persons[target].update({"body_points":first_human.get("body_points")})
    #persons[target].update({"feature":first_human.get("feature")})
    #persons[target].update({"last_update":count})
    return (target,error)
if __name__ == '__main__':
    PATH_TO_TEST_IMAGES_DIR = 'images/xahid_youya/test_openpose'
    TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.png'))
    count=0
    persons=[]
    color=[]
    #list.sort(TEST_IMAGE_PATHS)
    TEST_IMAGE_PATHS.sort(key=lambda f: int(filter(str.isdigit, f)))
    for im_file in TEST_IMAGE_PATHS:
        #im_file=list.sort(TEST_IMAGE_PATHS)[m]
        print(im_file)
        img = cv2.imread(im_file)
        #print(im_file)
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
        (image,human_info,human_boxes,feature) = TfPoseEstimator.draw_humans(im_file, image, humans, imgcopy=False)
        temp_info=feature_extraction(human_info,human_boxes,feature,count)
        #put the persons in the first frame in to a list
        if count==0:
            for i in range(len(temp_info)):
                color_person = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                color.append(color_person)
            persons=temp_info
            #cv2.imshow("test", image)
            #cv2.waitKey(1)
        #compare the persons in the nex frame with the persons in the previus frame(based on their body points and their features) and if they appears in the previus frame,update their body points and their bounding boxes;if not then added them as a new persons
        else:
            exclude=[]
            for i in range(len(temp_info)):
                (target,error)=find_min(temp_info[i],persons,count,exclude)
                print(error)
                if error>200:
                    persons.append({"id":len(persons),"body_points":temp_info[i].get("body_points"),"last_update":count,"bound_box":temp_info[i].get("boudn_boxes"),"feature":temp_info[i].get("feature")})
                        #exclude.append(target)
                        #(target,error)=find_min(temp_info[i],persons,count,exclude)
                else:
                    exclude.append(target)
                    print(exclude)
                    persons[target].update({"body_points":temp_info[i].get("body_points")})
                    persons[target].update({"bound_box":temp_info[i].get("bound_box")})
                    persons[target].update({"feature":temp_info[i].get("feature")})
                    persons[target].update({"last_update":count})
        (image,color)=draw_persons(image,count,persons,color)
        write_to_file(persons,im_file,count)
                #cv2.imshow("test", image)
                #cv2.waitKey(1)
        count=count+1
        im_file1=im_file.replace('images/xahid_youya/test_openpose','')
        #print(im_file1)
        op_imfile = 'images/xahid_youya/output'+im_file1
        #print(op_imfile)
        cv2.imwrite(op_imfile,image)
