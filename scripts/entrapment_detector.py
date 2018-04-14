#!/usr/bin/env python

"""
entrapment_detector.py
Detect the entrapment of the AutoKrawlers.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://mrsdprojects.ri.cmu.edu/2017teami/"
__copyright__   = "Copyright (C) 2018, the Moon Wreckers. All rights reserved."

import rospy
import numpy as np
from collections import deque
from std_msgs.msg import String, Float32, UInt16
from sensor_msgs.msg import Imu, JointState, Joy
from geometry_msgs.msg import Vector3Stamped, Twist
from nav_msgs.msg import Odometry
from DecayFilter import DecayFilter

import csv


CONFIG_ENABLE_LOG = False
CONFIG_DEBUG_PUBLISHER = False
CONFIG_DEBUG_PRINT_DATA_SOURCE = True

CONFIG_TOPIC_WHEELODOM = '/ak1/odom'
CONFIG_TOPIC_REFODOM = '/vive/LHR_0EB0243A_odom' # ak1: /vive/LHR_0EB0243A_odom, ak2: /vive/LHR_08DF7BFF_odom


ak_wheelodom = Odometry()
ak_refodom = Odometry()


def normpdf(x, mu, sigma2):
    return np.exp(- (x - mu)**2 / (2*sigma2)) / np.sqrt(2 * np.pi * sigma2)


def Pr_L_consistent(L):
    return normpdf(L, 0, 0.042947) * 2


def Pr_L_diverged(L):
    #return normpdf(L, 0.426055, 0.011208)
    return normpdf(L, 0.192272, 0.021208)


def Pr_v_stopped(v):
    return normpdf(v, 0, 0.000137) * 2


def Pr_v_moving(v):
    return normpdf(v, 0.252618, 0.022222)


def normalize(vec, protection=1e-4):
    vec = np.array(vec).astype(float)
    Z = 0
    for i in range(len(vec)):
        Z += vec[i,0]
    for i in range(len(vec)):
        vec[i,0] = min(max(vec[i,0]/Z, protection/float(len(vec)-1)), 1.0-protection)
    return vec


def cb_ak_wheelodom(data):
    global ak_wheelodom
    ak_wheelodom = data


def cb_ak_refodom(data):
    global ak_refodom
    ak_refodom = data


def entrapment_detector():
    if CONFIG_DEBUG_PUBLISHER:
        debug_prefix = '/debug/'
    else:
        debug_prefix = ''

    last_status_filtered = 'stopped'

    p_D = np.array([[0.01],  # D = diverged
                    [0.99]]) # D = consistent

    p_M = np.array([[0.01],  # M = moving
                    [0.99]]) # M = stopped

    # define feature weight matrix
    R = np.array([[1, 0],
                  [0, 0.05]])

    # define probability limit protection
    prob_protection = 1e-4

    # initialize odometries to listen to
    topic_wheelodom = CONFIG_TOPIC_WHEELODOM
    topic_refodom = CONFIG_TOPIC_REFODOM

    # initialize ROS node
    rospy.init_node('entrapment_detector', anonymous=True)

    # initialize subscribers
    rospy.Subscriber(topic_wheelodom, Odometry, cb_ak_wheelodom)
    rospy.Subscriber(topic_refodom, Odometry, cb_ak_refodom)

    # initialize publishers
    pub_entrapped = rospy.Publisher(debug_prefix + 'health/prob/entrapped', Float32, queue_size=10)
    pub_slipping = rospy.Publisher(debug_prefix + 'health/prob/slipping', Float32, queue_size=10)
    pub_stopped = rospy.Publisher(debug_prefix + 'health/prob/stopped', Float32, queue_size=10)
    pub_moving = rospy.Publisher(debug_prefix + 'health/prob/moving', Float32, queue_size=10)
    pub_status = rospy.Publisher(debug_prefix + 'health/status', String, queue_size=10)
    pub_status_filtered = rospy.Publisher(debug_prefix + 'health/status_filtered', String, queue_size=10)

    # initialize log file writer
    if CONFIG_ENABLE_LOG:
        csvfile = open('entrapment_detection_log.csv', 'wb')
        spamwriter = csv.writer(
            csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    # initialize decay filters
    filterX = DecayFilter(d=0.95, wsize=32)
    filterY = DecayFilter(d=0.95, wsize=32)

    # loop process
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # receive new measurements
        X_raw = np.array([[ak_wheelodom.twist.twist.linear.x],
                          [ak_wheelodom.twist.twist.linear.y],
                          [ak_wheelodom.twist.twist.angular.z]])
        X_norm = np.array([[np.linalg.norm(X_raw[0:2])],
                          [X_raw[2,0]]])
        filterX.append(X_norm)

        Y_raw = np.array([[ak_refodom.twist.twist.linear.x],
                          [ak_refodom.twist.twist.linear.y],
                          [ak_refodom.twist.twist.angular.z]])
        Y_norm = np.array([[np.linalg.norm(Y_raw[0:2])],
                          [Y_raw[2,0]]])
        filterY.append(Y_norm)

        # filter measurements
        X = filterX.filtered
        Y = filterY.filtered

        # compute odometry divergence
        e = X - Y
        L = np.sqrt(np.dot(np.matrix.transpose(e), np.dot(R, e)))

        # update divergence status
        n_D = np.array([[Pr_L_diverged(L) * p_D[0,0]],
                        [Pr_L_consistent(L) * p_D[1,0]]])
        p_D = normalize(n_D, protection=prob_protection)

        # update movement status
        n_M = np.array([[Pr_v_moving(Y[0,0]) * p_M[0,0]],
                        [Pr_v_stopped(Y[0,0]) * p_M[1,0]]])
        p_M = normalize(n_M, protection=prob_protection)

        # compute health status likelihoods
        p_health = [p_D[0,0] * p_M[1,0], # entrapped
                    p_D[0,0] * p_M[0,0], # slipping
                    p_D[1,0] * p_M[1,0], # stopped
                    p_D[1,0] * p_M[0,0]] # moving

        # publish health status probabilities
        pub_entrapped.publish(p_health[0])
        pub_slipping.publish(p_health[1])
        pub_stopped.publish(p_health[2])
        pub_moving.publish(p_health[3])

        # publish health status
        status_code = np.argmax(p_health)
        if status_code == 0:
            pub_status.publish('entrapped')
            last_status_filtered = 'entrapped'
        elif status_code == 1:
            pub_status.publish('slipping')
            last_status_filtered = 'slipping'
        elif status_code == 2:
            pub_status.publish('stopped')
            if last_status_filtered != 'entrapped':
                last_status_filtered = 'stopped'
        elif status_code == 3:
            pub_status.publish('moving')
            last_status_filtered = 'moving'
        else:
            rospy.logerr('unexpected health status code (%s)' % (status_code))

        pub_status_filtered.publish(last_status_filtered)

        # print log
        if CONFIG_DEBUG_PRINT_DATA_SOURCE:
            print('X=[%f, %f], Y=[%f, %f], L=%f, S=%s' % (X[0,0], X[1,0], Y[0,0], Y[1,0], L, last_status_filtered))
        else:
            print('L=%s, v=%s, P(dvg)=%s, P(stp)=%s, P(entr)=%s, status_code=%s' % (
                float(L), float(Y[0,0]), float(p_D[0,0]), float(p_M[1,0]), float(p_health[0]), status_code
            ))

        # write log file
        if CONFIG_ENABLE_LOG:
            output_line = [
                float(L),
                float(Y[0,0]),
                float(p_D[0,0]),
                float(p_D[1,0]),
                float(p_M[0,0]),
                float(p_M[1,0]),
                float(p_health[0]),
                float(p_health[1]),
                float(p_health[2]),
                float(p_health[3]),
            ]
            spamwriter.writerow(output_line)

        # frequency control
        rate.sleep()


if __name__ == '__main__':
    try:
        entrapment_detector()
    except rospy.ROSInterruptException:
        pass
