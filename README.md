### tf-pose-estimation

TensorFlow implementation of OpenPose; forked from https://github.com/ildoonet/tf-pose-estimation

### Install
Assuming Python 2.7, OpenCV 3, and TensorFlow 1.4+ are already installed:
- pip install -r requirements.txt
- sudo apt install swig
- sudo pip install face_recognition
- cd tf_pose/pafprocess/
- swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

> if you need the package installization and full functionalities, follow the installization processes in the original repo

#### References

[0] https://github.com/ildoonet/tf-pose-estimation
[1] https://github.com/CMU-Perceptual-Computing-Lab/openpose
[2] https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation


