#!/usr/bin/env python

"""
entrapment_generator.py
AutoKrawlers entrapment data generator.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://mrsdprojects.ri.cmu.edu/2017teami/"
__copyright__   = "Copyright (C) 2018, the Moon Wreckers. All rights reserved."

import rospy
import csv
import numpy as np
from std_msgs.msg import String, Float32, UInt16
from sensor_msgs.msg import Imu, JointState, Joy
from geometry_msgs.msg import Vector3Stamped, Twist
from nav_msgs.msg import Odometry


ak_wheelodom = Odometry()
ak_refodom = Odometry()


def cb_ak_wheelodom(data):
    global ak_wheelodom
    ak_wheelodom = data


def cb_ak_refodom(data):
    global ak_refodom
    ak_refodom = data


def entrapment_generator():
    topic_wheelodom = '/ak1/odom'
    topic_refodom = '/vive/LHR_0EB0243A_odom'

    R = np.array([[1, 0],
                  [0, 0.05]])

    rospy.init_node('entrapment_generator', anonymous=True)

    rospy.Subscriber(topic_wheelodom, Odometry, cb_ak_wheelodom)
    rospy.Subscriber(topic_refodom, Odometry, cb_ak_refodom)

    csvfile = open('entrapment_data.csv', 'wb')
    spamwriter = csv.writer(
        csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():

        X_raw = np.array([[ak_wheelodom.twist.twist.linear.x],
                          [ak_wheelodom.twist.twist.linear.y],
                          [ak_wheelodom.twist.twist.angular.z]])
        X = np.array([[np.linalg.norm(X_raw[0:2])],
                      [X_raw[2,0]]])

        Y_raw = np.array([[ak_refodom.twist.twist.linear.x],
                          [ak_refodom.twist.twist.linear.y],
                          [ak_refodom.twist.twist.angular.z]])
        Y = np.array([[np.linalg.norm(Y_raw[0:2])],
                      [Y_raw[2,0]]])

        e = X - Y

        l = np.sqrt(np.dot(np.matrix.transpose(e), np.dot(R, e)))

        output_line = [
            X[0,0], X[1,0], Y[0,0], Y[1,0], l[0,0]
        ]
        spamwriter.writerow(output_line)

        rate.sleep()


if __name__ == '__main__':
    try:
        entrapment_generator()
    except rospy.ROSInterruptException:
        pass
