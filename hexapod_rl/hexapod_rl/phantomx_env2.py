#!/usr/bin/env python3
import rospy
import gymnasium as gym
import numpy as np
import math
import cv2
from gymnasium import spaces
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

class PhantomXEnv(gym.Env):
    def __init__(self):
        # rospy.init_node("phantomx_env_node", anonymous=True)
        self.imu_data = None
        self.imu_sub = rospy.Subscriber("/phantomx/imu/data", Imu, self._imu_cb, queue_size=1)
        self.cmd_pub = rospy.Publisher("/phantomx/cmd_vel", Twist, queue_size=1)
        self.model_state_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_state_cb)
        self.set_model_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.episode_start_time = rospy.Time.now()
        self.bridge = CvBridge()
        self.last_image = None
        self.safe_x_bounds = (-2.5, 2.5)
        self.safe_y_bounds = (-2.5, 2.5)

        self.image_sub = rospy.Subscriber("/phantomx/camera/image_raw", Image, self._image_cb)

        self.prev_dist_rb = None
        self.prev_dist_bt = None
        self.phase = 0
        self.find_threshold = 0.5
        self.goal_threshold = 0.5
        self.target_pos = np.array([2.0, 2.0])
        self.visual_goal_initialized = False

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
        })

        rospy.sleep(1.0)

    def _image_cb(self, msg):
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            rospy.logwarn(f"Image conversion failed: {e}")

    def _imu_cb(self, msg: Imu):
        self.imu_data = msg

    def _model_state_cb(self, msg):
        self.model_names = msg.name
        try:
            self.robot_pose = msg.pose[msg.name.index("phantomx")]
            self.block_pose = msg.pose[msg.name.index("push_block")]
        except ValueError:
            pass

    def _is_out_of_bounds(self, xy):
        x, y = xy
        return not (self.safe_x_bounds[0] <= x <= self.safe_x_bounds[1] and
                    self.safe_y_bounds[0] <= y <= self.safe_y_bounds[1])

    def step(self, action):
        twist = Twist()
        if action == 0:
            twist.linear.x = 2.0
        elif action == 1:
            twist.angular.z = 1.5
        elif action == 2:
            twist.angular.z = -1.5
        self.cmd_pub.publish(twist)
        rospy.sleep(0.05)

        obs = self._get_obs()
        vec = obs["vector"]
        robot_xy = vec[0:2]

        reward, done = self._compute_reward(obs)
        terminated = done
        truncated = False

        elapsed = (rospy.Time.now() - self.episode_start_time).to_sec()
        if elapsed > 60:
            rospy.logwarn("Episode truncated: timeout reached!")
            reward -= 5.0
            truncated = True
            terminated = False
            self.cmd_pub.publish(Twist()); rospy.sleep(0.05)

        if self._is_out_of_bounds(robot_xy):
            rospy.logwarn("Robot hit wall or exited safe area!")
            reward -= 10.0
            terminated = True
            truncated = False
            self.cmd_pub.publish(Twist()); rospy.sleep(0.05)

        info = {"phase": self.phase,
                "dist_rb": float(np.linalg.norm(vec[2:4] - robot_xy)),
                "dist_bt": float(np.linalg.norm(vec[2:4] - vec[4:6])),
                "wall_crash": self._is_out_of_bounds(robot_xy)}

        if terminated or truncated:
            new_obs, new_info = self.reset()
            new_info.update(info)
            return new_obs, float(reward), terminated, truncated, new_info

        return obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # rospy.wait_for_service('/gazebo/pause_physics')
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        # if not self.visual_goal_initialized:
        #     rospy.loginfo("pause_physics")
        #     pause_physics()
        #     self._set_model_pose("target_marker", 2.0, 2.0, 0.2)
        #     unpause_physics()
        #     self.target_pos = np.array([2.0, 2.0])
        #     self.visual_goal_initialized = True

        self._set_model_pose("phantomx", 0.0, 0.0, 0.25)
        x = np.random.uniform(0.7, 1.7)
        y = np.random.uniform(0.7, 1.7)
        self._set_model_pose("push_block", x, y, 0.2)

        self.prev_dist_rb = None
        self.prev_dist_bt = None
        self.phase = 0
        self.episode_start_time = rospy.Time.now()
        rospy.sleep(0.05)

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # build vector
        base = np.zeros(6, dtype=np.float32)
        yaw = 0.0
        yaw_error_block = 0.0
        yaw_error_goal = 0.0

        if hasattr(self, 'robot_pose') and self.robot_pose and self.block_pose:
            r = self.robot_pose.position
            b = self.block_pose.position
            robot_xy = np.array([r.x, r.y], dtype=np.float32)
            block_xy = np.array([b.x, b.y], dtype=np.float32)
            goal_xy  = self.target_pos.astype(np.float32)
            base = np.concatenate([robot_xy, block_xy, goal_xy])

            q = self.robot_pose.orientation
            yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

            angle_block = math.atan2(block_xy[1]-robot_xy[1], block_xy[0]-robot_xy[0])
            yaw_error_block = (angle_block - yaw + np.pi) % (2*np.pi) - np.pi
            angle_goal  = math.atan2(goal_xy[1]-block_xy[1], goal_xy[0]-block_xy[0])
            yaw_error_goal  = (angle_goal  - yaw + np.pi) % (2*np.pi) - np.pi

        imu_arr = np.zeros(6, dtype=np.float32)
        if self.imu_data:
            la, av = self.imu_data.linear_acceleration, self.imu_data.angular_velocity
            imu_arr = np.array([la.x, la.y, la.z, av.x, av.y, av.z], dtype=np.float32)

        vector = np.concatenate([base,
                                 np.array([yaw, yaw_error_block, yaw_error_goal], dtype=np.float32),
                                 imu_arr])
        # ensure float32
        vector = vector.astype(np.float32)

        # build image as uint8
        if self.last_image is not None:
            img = cv2.resize(self.last_image, (64, 64)).astype(np.float32) / 255.0
        else:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
        img = img.transpose(2, 0, 1)  # CHW

        return {"vector": vector, "image": img}

    def _compute_reward(self, obs):
        vec = obs["vector"]
        img = obs["image"]
        robot_xy = vec[0:2]
        block_xy = vec[2:4]
        goal_xy  = vec[4:6]
        yaw_error_block = vec[7]
        yaw_error_goal  = vec[8]

        dist_rb = np.linalg.norm(block_xy - robot_xy)
        dist_bt = np.linalg.norm(block_xy - goal_xy)

        reward = 0.0
        done = False
        eps = 0.01

        if self.phase == 0:
            reward -= dist_rb
            reward += self.camera_block_visibility_reward(img) * 5.0
            # rospy.logwarn(f"camera reward: {self.camera_block_visibility_reward(img) * 5.0}")
            if self.prev_dist_rb is not None:
                delta = self.prev_dist_rb - dist_rb
                if   delta > eps:  reward += delta * 10.0
                elif delta < -eps: reward += delta * 15.0
            self.prev_dist_rb = dist_rb

            if dist_rb <= self.find_threshold:
                reward += 40.0  # intermediate bonus
                self.phase = 1
                rospy.logwarn("Block found!")

        else:
            reward -= dist_bt
            reward += self.centering_reward(img) * 5.0
            # rospy.logwarn(f"centering reward: {self.centering_reward(img) * 5.0}")
            err_b = min(abs(yaw_error_block),    np.pi - abs(yaw_error_block))
            err_g = min(abs(yaw_error_block-yaw_error_goal), np.pi - abs(yaw_error_block-yaw_error_goal))
            reward += (1 - err_b/np.pi) + (1 - err_g/np.pi)
            if self.prev_dist_bt is not None:
                delta = self.prev_dist_bt - dist_bt
                if   delta > eps:  reward += delta * 20.0
                elif delta < -eps: reward += delta * 30.0
            self.prev_dist_bt = dist_bt

            if dist_bt <= self.goal_threshold:
                reward += 100.0
                done = True
                rospy.logwarn("Goal reached!")

        reward -= 0.1
        return float(reward), done

    def camera_block_visibility_reward(self, img):
        img = img.transpose(1,2,0)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array([100,150,50])
        upper = np.array([140,255,255])
        mask = cv2.inRange(hsv, lower, upper)
        return np.sum(mask)/255.0/(64*64)

    def centering_reward(self, img):
        img = img.transpose(1,2,0)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array([100,150,50])
        upper = np.array([140,255,255])
        mask = cv2.inRange(hsv, lower, upper)
        M = cv2.moments(mask)
        if M["m00"]>0:
            cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
            d = np.linalg.norm([cx-32, cy-32])
            return 1 - d/np.linalg.norm([32,32])
        return 0.0

    def _set_model_pose(self, name, x, y, z):
        msg = ModelState()
        msg.model_name = name
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        msg.twist = Twist()
        msg.reference_frame = "world"
        try:
            self.set_model_srv(msg)
        except rospy.ServiceException as e:
            rospy.logwarn(f"Failed to set pose for {name}: {e}")