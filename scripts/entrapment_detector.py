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
from std_msgs.msg import String, Float32, UInt16
from sensor_msgs.msg import Imu, JointState, Joy
from geometry_msgs.msg import Vector3Stamped, Twist
from nav_msgs.msg import Odometry


ak_wheelodom = Odometry()
ak_refodom = Odometry()


def normpdf(x, mu, sigma2):
    return np.exp(- (x - mu)**2 / (2*sigma2)) / np.sqrt(2 * np.pi * sigma2)


def Pr_L_consistent(L):
    return normpdf(L, 0, 0.042947) * 2


def Pr_L_diverged(L):
    return normpdf(L, 0.426055, 0.011208)


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
    topic_wheelodom = '/ak1/odom'
    topic_refodom = '/vive/LHR_0EB0243A_odom'

    R = np.array([[1, 0],
                  [0, 0.05]])

    rospy.init_node('entrapment_detector', anonymous=True)

    rospy.Subscriber(topic_wheelodom, Odometry, cb_ak_wheelodom)
    rospy.Subscriber(topic_refodom, Odometry, cb_ak_refodom)

    p_D = np.array([[0.01],  # D = diverged
                    [0.99]]) # D = consistent

    p_M = np.array([[0.01],  # M = moving
                    [0.99]]) # M = stopped

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

        L = np.sqrt(np.dot(np.matrix.transpose(e), np.dot(R, e)))

        # update divergence status
        n_D = np.array([[Pr_L_diverged(L) * p_D[0,0]],
                        [Pr_L_consistent(L) * p_D[1,0]]])
        p_D = normalize(n_D)

        # update movement status
        n_M = np.array([[Pr_v_moving(Y[0,0]) * p_M[0,0]],
                        [Pr_v_stopped(Y[0,0]) * p_M[1,0]]])
        p_M = normalize(n_M)

        # compute entrapment likelihood
        p_entrapped = p_D[0,0] * p_M[1,0]

        #print('%.6s => %.6s, %.6s' % (Y[0,0], Pr_v_moving(Y[0,0]), Pr_v_stopped(Y[0,0])))
        print('P(dvg)=%s, P(stp)=%s, P(entr)=%s' % (
            p_D[0,0], p_M[1,0], p_entrapped
        ))

        rate.sleep()


if __name__ == '__main__':
    try:
        entrapment_detector()
    except rospy.ROSInterruptException:
        pass
