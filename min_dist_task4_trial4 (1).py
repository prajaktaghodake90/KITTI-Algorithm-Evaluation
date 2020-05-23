#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
from math import cos, sin, pi, floor, sqrt
import ast
import os

def callback(msg):

    #task 1
	#raw data from the RPLidar
	scan_data = msg.ranges
	min_ang = msg.angle_min
	incr_ang = msg.angle_increment
	print "----------------------------------------------------obj1----------"
	#print index and dist of closest point of object
	(index_min,dist_min)= get_index_of_closest_point(scan_data)
	print "\n","Minimal distance to the centre of lidar is:",dist_min*100," cm"
	
    #task 2
	#print angle of closest point
	angle_min=get_angle_at_index(scan_data,incr_ang,index_min)
	print "\n","Angular position of object in degrees is:",angle_min*180/pi

    #task 3
	#print length and width of object
	get_side1_side2(scan_data,index_min,incr_ang,dist_min)     
        
                   
    #task 4
        cnt=0
        obj=1
        for i in range (len(scan_data)):
            if scan_data[i]== ' inf':
                cnt=cnt+1
        if cnt < len(scan_data):
            obj=obj+1
            lst=list(scan_data)
            for i in range(index_side2,index_side1+1):
                lst[i]=0.0
            scan_data=tuple(lst)
            print "---------------------------------------------",obj,"-------------------"
            (index_min,dist_min)= get_index_of_closest_point(scan_data)
            print "\n","Minimal distance to the centre of lidar is:",dist_min*100," cm"
            angle_min=get_angle_at_index(scan_data,incr_ang,index_min)
            print "\n","Angular position of object in degrees is:",angle_min*180/pi
            get_side1_side2(scan_data,index_min,incr_ang,dist_min)
            print "---------------------------------------------end of ",obj,"--------------------------------------------"
        else:
            print "all objects detected"
'''
TODO: find the index and dist of the closest point in the scan_data
'''
def get_index_of_closest_point(scan_data):
	dist_closest_point = 12.0
	index_closest_point =0
        for i in range (len(scan_data)):
		if (scan_data[i] > 0.0 ) and (scan_data[i] < dist_closest_point):
			dist_closest_point = scan_data[i]
			index_closest_point = i
			
        return (index_closest_point,dist_closest_point)

'''
TODO: calculate the angle in rad for the closest point in scan_data
'''

def get_angle_at_index(scan_data,incr_ang,index):
	#angle = min_ang+(index*incr_ang)
        angle = index*incr_ang
        return angle

'''
TODO: calculate index and dist of side 1
'''
def get_index_dist_side1(scan_data,index_min):
        i_side1=0
    	for i in range (index_min+1,len(scan_data)):
            i_side1=i_side1+1
            if(scan_data[i]==float(' inf')):
                break
        
        index_side1=index_min+i_side1-1
        #print(index_min+i_side1-1)
        dist_side1 = scan_data[index_side1]
        #print('side 1 distance',dist_side1*100)
        return (index_side1,dist_side1)
'''
TODO: calculate index and dist of side 2
'''        
def get_index_dist_side2(scan_data,index_min,dist_min):
        i_side2=0
	for i in range (index_min,1,-1):
            i_side2=i_side2+1
            if scan_data[i]==float(' inf'):
                break
        index_side2=index_min-i_side2    
        dist_side2 = scan_data[index_side2+2]
        #print('side 1 distance',dist_side2*100)
        return (index_side2,dist_side2)

def get_side1_side2(scan_data,index_min,incr_ang,dist_min):
    (index_side1,dist_side1)=get_index_dist_side1(scan_data,index_min)
    angle_side1=get_angle_at_index(scan_data,incr_ang,index_side1)
    index_s1_min=index_side1-index_min
    angle_side1_min= get_angle_at_index(scan_data,incr_ang,index_s1_min)
    side1= sqrt(pow(dist_min,2.0)+pow(dist_side1,2.0)-(2*dist_min*dist_side1*cos(angle_side1_min)))
    #print "\n","length of side1 is:",side1*100
            
    (index_side2,dist_side2)=get_index_dist_side2(scan_data,index_min,dist_min)
    angle_side2=get_angle_at_index(scan_data,incr_ang,index_side2)
    index_min_s2= index_min-index_side2
    angle_min_side2= get_angle_at_index(scan_data,incr_ang,index_min_s2)
    side2= sqrt(pow(dist_min,2.0)+pow(dist_side2,2.0)-(2*dist_min*dist_side2*cos(angle_min_side2)))
    #print "\n","length of side2:",side2*100

    if abs(dist_side1-dist_side2)>0.02:
            if dist_side1>dist_side2:
                print "L shaped object detected with width:",side1*100," cm"," and length:",side2," cm"
            else:
                print "L shaped object detected with width:",side2*100," cm"," and length:",side1," cm"
        else:
            print "I shaped object detected with length:",(side1+side2)*100," cm"
        
    return
  
rospy.init_node('dist_min')
rospy.Subscriber('/scan',LaserScan, callback)
rospy.spin()
