# KITTI-Algorithm-Evaluation
Evaluated a birds-eye view (BEV) object detector on the KITTI dataset
1. Write a converter:
The KITTI lidar data has converted into a data format used by the algorithms. Provide this converter for the lidar data and the bounding boxes.
In addition to that, a visualization would also be helpful to debug your code.
2. Centre point comparison:
Expand predict.py file to check the centre point if the detection is correct. A single detection is considered to be correct if the distance between the centre point of the detected bounding box and the centre point of the groundtruth bounding box is below 50 pixels.
Count your true positives and false positives and provide the average precision.
3. IoU comparison:
Expand predict.py file to use the intersection over union (IoU) to check if the results are to be considered correct. A single detection is considered to be correct if the IoU is above or equal to 0.7.
Again, true positives and false positives and provide the average precision are counted.
