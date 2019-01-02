### tf-pose-estimation

TensorFlow implementation of OpenPose; forked from https://github.com/ildoonet/tf-pose-estimation

### Install
- pip install -r requirements.txt
- sudo apt install swig
- sudo pip install face_recognition
- cd tf_pose/pafprocess/
- swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

>> Note that this repo has reduced functionalities; if you need the full package and other modules, follow the installization process in the original repo

#### References

[0] https://github.com/ildoonet/tf-pose-estimation

[1] https://github.com/CMU-Perceptual-Computing-Lab/openpose

[2] https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation


