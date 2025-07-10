#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def main():
    rospy.init_node('camera_viewer_node', anonymous=True)
    bridge = CvBridge()

    def image_callback(msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imshow("PhantomX Camera View", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    # Subscribe to the camera topic
    rospy.Subscriber("/phantomx/camera/image_raw", Image, image_callback)

    rospy.logwarn("Camera viewer started. Press Ctrl+C to exit.")
    rospy.spin()

    # Clean shutdown
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
