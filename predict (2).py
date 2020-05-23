"""
    Predict and draw bounding boxes on images using loaded model. 
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import utils
import imageio
import shapely.geometry
import shapely.affinity
import matplotlib.pyplot as plt

result_matrix = []
final_predicted_matrix = []
number_predicted = 0
number_groundtruth = 0

# Load json configs
with open('config.json', 'r') as f:
    config = json.load(f)
boundary = config["boundary"]

def eval(model_dir, idx):
    """
        model_dir: path to the file where the model weights are saved (ModelWeights.pth)
        idx: path to the file where the indices are saved (eval.txt), you can use a loop over all indices
    
    """
    '''***************Code changes done for center point starts*****************'''
    groundtruth_box_rotated = []
    predicted_box_rotated = []
    global number_predicted
    global number_groundtruth
    '''***************Code changes done for center point ends*****************'''
    
    # Load point cloud data
    
    lidar_file = 'C:/Lidar_kittiDataset/Training/velodyne/'+str(idx)+'.bin'  # path to one lidar file with an indice from idx (e.g. Path/to/file/velodyne/003001.bin)
    calibration_file = 'C:/Lidar_kittiDataset/Training/calib/'+str(idx) +'.txt' # path to one calibration file with an indice from idx (e.g. Path/to/file/calib/003001.txt)
    label_file = 'C:/Lidar_kittiDataset/Training/label_2/'+str(idx)+'.txt' # path to the ground truth label file with an indice from idx (e.g. Path/to/file/label_2/003001.txt)
    
#    lidar_file = 'C:/Lidar_kittiDataset/Training/velodyne/003001.bin'  # path to one lidar file with an indice from idx (e.g. Path/to/file/velodyne/003001.bin)
#    calibration_file = 'C:/Lidar_kittiDataset/Training/calib/003001.txt'# path to one calibration file with an indice from idx (e.g. Path/to/file/calib/003001.txt)
#    label_file = 'C:/Lidar_kittiDataset/Training/label_2/003001.txt'
    calib = utils.calibration(calibration_file)
    target = utils.get_target(label_file, calib)
    
    model = torch.load(model_dir) # model that calculates a matrix from input, so that with get_region_boxes boxes can be calculated
    model.cpu()     # says that the cpu should be used
    
    a = np.fromfile(lidar_file, dtype = np.float32).reshape(-1, 4) # raw point cloud
    b = utils.removePoints(a, boundary)     # deletes all points that are not in the boundary
    rgb_map = utils.makeBVFeature(b, boundary, 40 / 512) # # creates the BEV represantation
    # Load trained model and forward, raw input (512, 1024, 3)
    input = torch.from_numpy(rgb_map)       # convertes the numpy array into a torch tensor
    input = input.reshape(1, 3, 512, 1024)  # reshape the tensor, so that he has the right shape
    
    img = rgb_map.copy()
    img = (img - img.min()) * 255/ (img.max() - img.min()) # normalize the values so that they are between 0 and 255
    img = img.astype(np.uint8)  # change data type
    for j in range(len(target)):
        if target[j][1] == 0:
            break
        ground_truth_x = int(target[j][1] * 1024.0)
        ground_truth_y = int(target[j][2] * 512.0)
        ground_truth_width  = int(target[j][3] * 1024.0)
        ground_truth_length = int(target[j][4] * 512.0)
        rect_top1 = int(ground_truth_x - ground_truth_width / 2)
        rect_top2 = int(ground_truth_y - ground_truth_length / 2)
        rect_bottom1 = int(ground_truth_x + ground_truth_width / 2)
        rect_bottom2 = int(ground_truth_y + ground_truth_length / 2)
#        cv2.rectangle(img, (rect_top1, rect_top2), (rect_bottom1, rect_bottom2), (0, 0, 255), 1) # Here the boxes are drawn without the rotation
                                                                                                 # For your evaluation with IoU you could write 
                                                                                                 # a class for rotated rectangles. 
                                                                                                 # For this the libraty shapely and especially 
                                                                                                 # shapely.geometry.box (creating a box)
                                                                                                 # shapely.affinity.rotate and shapely.affinity.translate 
                                                                                                 # (rotating and translating) the box
                                                                                                 # Also the intersection of two such rectangles can be calculated
                                                                                                 # pretty easy.
                                                                                                 # For the comparison of the box center this is actual not
                                                                                                 # needed, but can also be used
        # Example:
        angle = -1 * np.arctan2(target[j][6], target[j][5])
        gt_bbox = shapely.geometry.box(rect_top1,rect_top2, rect_bottom1,rect_bottom2)
        gt_bbox = shapely.affinity.rotate(gt_bbox, angle, use_radians=True)
        corners = gt_bbox.exterior.coords[:]
        groundtruth_box_rotated.append([gt_bbox])
        
#        for j in range(len(corners)):
#            cv2.line(img, (int(corners[j][0]), int(corners[j][1])),
#                     (int(corners[(j + 1) % len(corners)][0]), int(corners[(j + 1) % len(corners)][1])),
#                     (255, 0, 255), 1)
                                                
#        list_of_groundtruth_cp.append([ground_truth_x,ground_truth_y])                                      
    # Set model mode to determine whether batch normalization and dropout are engaged
    model.eval()

    output = model(input.float())

    all_boxes = utils.get_region_boxes(output)# saves an image of the BEV with the boxes
    
    
    for i in range(len(all_boxes)):
#        print("Box predicted!") 
        
        pred_x = int(all_boxes[i][0] * 1024.0 / 32.0)
        pred_y = int(all_boxes[i][1] * 512.0 / 16.0)
        pred_width = int(all_boxes[i][2] * 1024.0 / 32.0)
        pred_length = int(all_boxes[i][3] * 512.0 / 16.0)  

        rect_top1 = int(pred_x - pred_width / 2)
        rect_top2 = int(pred_y - pred_length / 2)
        rect_bottom1 = int(pred_x + pred_width / 2)
        rect_bottom2 = int(pred_y + pred_length / 2)
#        cv2.rectangle(img, (rect_top1, rect_top2), (rect_bottom1, rect_bottom2), (0, 255, 0), 1)
        
        pre_angle = -1 * np.arctan2(all_boxes[i][4], all_boxes[i][5])
        pre_bbox = shapely.geometry.box(rect_top1,rect_top2, rect_bottom1,rect_bottom2)
        pre_bbox = shapely.affinity.rotate(pre_bbox, pre_angle, use_radians=True)
        corners_pre = pre_bbox.exterior.coords[:]
        predicted_box_rotated.append([pre_bbox])
#        my_corners_pre.append(corners_pre)
#        for j in range(len(corners_pre)):
#            cv2.line(img, (int(corners_pre[j][0]), int(corners_pre[j][1])),
#                     (int(corners_pre[(j + 1) % len(corners_pre)][0]), int(corners_pre[(j + 1) % len(corners_pre)][1])),
#                     (0, 0, 0), 1)   
            
#        list_of_predict_cp.append([pred_x,pred_y])
    utils.cp_and_avg_precision(target,all_boxes,idx,result_matrix)
    
    number_groundtruth = number_groundtruth+len(target)
    number_predicted = number_predicted+len(all_boxes)
#    imageio.imwrite('predict/eval_bv.png', img)

if __name__ == "__main__":
    with open("eval.txt", "r") as f:
        image_idx = f.readlines()
    idx = [int(line.rstrip("\n")) for line in image_idx]
    model_dir = './ModelWeights.pth'

    # Load model, run predictions and draw boxes
    for i in range (0,len(idx)-1497):
        print('for idx:',idx[i])
        eval(model_dir,"%06d" %idx[i])
        
#print(result_matrix)
result_matrix_sorted = sorted(result_matrix, key=lambda x:x[2], reverse=True)
#print('result_matrix_sorted',result_matrix_sorted)
#print('Number of predicted boxes',len(result_matrix_sorted))

for i in range(len(result_matrix_sorted)):
    if 'TP' in result_matrix_sorted[i]:
        final_predicted_matrix.append([result_matrix_sorted[i][0],result_matrix_sorted[i][1],result_matrix_sorted[i][2],1,0])
    elif 'FP' in result_matrix_sorted[i]:
        final_predicted_matrix.append([result_matrix_sorted[i][0],result_matrix_sorted[i][1],result_matrix_sorted[i][2],0,1])
#        print('FK',result_matrix_sorted[i])

cum_tp = 0
cum_fp = 0
for i in range(len(final_predicted_matrix)):
#    print(final_predicted_matrix[i])
    if final_predicted_matrix[i][3]==1 and final_predicted_matrix[i][4]==0:
        cum_tp=cum_tp+1
    elif final_predicted_matrix[i][3]==0 and final_predicted_matrix[i][4]==1:
        cum_fp=cum_fp+1
    final_predicted_matrix[i].append(cum_tp)
    final_predicted_matrix[i].append(cum_fp)
    
#    print(final_predicted_matrix[i])
    
for i in range(len(final_predicted_matrix)):
#    precision = final_predicted_matrix[i][5]/number_predicted
    precision = final_predicted_matrix[i][5]/(final_predicted_matrix[i][5]+final_predicted_matrix[i][6])
    recall = final_predicted_matrix[i][5]/number_groundtruth
    final_predicted_matrix[i].append(precision)
    final_predicted_matrix[i].append(recall)
    
final_predicted_matrix.insert(0,['Image id','predicted box index','score','TP','FP','acc TP','acc FP','precision','recall'])
precision_list=[]
recall_list=[]
for i in range(1,len(final_predicted_matrix)):
#    print(final_predicted_matrix[i])
    precision_list.append(final_predicted_matrix[i][7])
    recall_list.append(final_predicted_matrix[i][8])
    
plt.plot(recall_list, precision_list)
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()

interpol_points = [0]
[interpol_points.append(r) for r in recall_list]
#nterpol_points.append(1)

precision_interpolated_list = [0]
[precision_interpolated_list.append(e) for e in precision_list]
#recision_interpolated_list.append(0)


for i in range(len(precision_interpolated_list) - 1, 0, -1):
    precision_interpolated_list[i - 1] = max(precision_interpolated_list[i - 1], precision_interpolated_list[i])

plt.plot(recall_list, precision_list)
plt.plot(interpol_points, precision_interpolated_list, '--r', label='11-point interpolated precision')
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()

ap = 0
for i in range(1,len(precision_interpolated_list)):
    if interpol_points[i] - interpol_points[i - 1] != 0:
        ap += (interpol_points[i] - interpol_points[i - 1]) * precision_interpolated_list[i]

print('ap',ap*100)
