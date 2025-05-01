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
from tf.transformations import euler_from_quaternion

class PhantomXEnv(gym.Env):
    def __init__(self):
        # rospy.init_node("phantomx_env_node", anonymous=True)
        self.imu_data = None
        self.imu_sub = rospy.Subscriber("/phantomx/imu/data",Imu,self._imu_cb,queue_size=1)
        self.cmd_pub = rospy.Publisher("/phantomx/cmd_vel", Twist, queue_size=1)
        self.model_state_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_state_cb)
        self.set_model_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.episode_start_time = rospy.Time.now()
        self.bridge = CvBridge()
        self.last_image = None
        self.safe_x_bounds = (-2.5, 2.5)
        self.safe_y_bounds = (-2.5, 2.5)
     

        self.image_sub = rospy.Subscriber("/phantomx/camera/image_raw", Image, self._image_cb)

        # initialize cumulative reward counter
        self.episode_reward = 0.0
        self.prev_dist_bt   = None
        self.prev_dist_rb = None
        self.model_names = None
        self.robot_pose = None
        self.block_pose = None
        self.target_pos = np.array([2.0, 2.0])  #goal

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32),
            "image": spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
        })


        self.phase = 0
        self.find_threshold = 0.3 # meters: when robot is “at” the block
        self.goal_threshold = 0.3  # meters: when block is “at” the goal

        rospy.sleep(1.0)  # ensure subscriptions are active

    
    def _image_cb(self, msg):
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            rospy.logwarn(f"Image conversion failed: {e}")
            
    def _imu_cb(self, msg: Imu):
        self.imu_data = msg

    def _is_out_of_bounds(self, robot_xy):
        """Helper: Check if robot is outside safe 5x5 meter box."""
        x, y = robot_xy
        return not (self.safe_x_bounds[0] <= x <= self.safe_x_bounds[1]
                    and self.safe_y_bounds[0] <= y <= self.safe_y_bounds[1])

    def _model_state_cb(self, msg):
        self.model_names = msg.name
        try:
            self.robot_pose = msg.pose[msg.name.index("phantomx")]
            self.block_pose = msg.pose[msg.name.index("push_block")]
        except ValueError:
            pass

    # def step(self, action):
    #     # ——— Your existing action → obs → reward logic ———
    #     twist = Twist()
    #     if action == 0:
    #         twist.linear.x = 1.0
    #     elif action == 1:
    #         twist.angular.z = 0.5
    #     elif action == 2:
    #         twist.angular.z = -0.5
    #     # action == 3 → stop (zero twist)
    #     self.cmd_pub.publish(twist)
    #     rospy.sleep(0.05)

    #     obs = self._get_obs()
    #     img = obs["image"]
    #     vec = obs["vector"]
    #     robot_xy = vec[0:2]

    #     # Initialize flags
    #     terminated = False
    #     truncated = False

    #     # Base reward + compute terminated flag if your _compute_reward returns it
    #     reward, term_flag = self._compute_reward(obs)
    #     # If your compute_reward yields a termination, capture it:
    #     terminated = terminated or term_flag

    #     # Camera bonuses, distance shaping, timeout, wall checks...
    #     # [Keep your existing reward shaping here]
    #     elapsed = (rospy.Time.now() - self.episode_start_time).to_sec()
    #     if elapsed > 60:
    #         truncated = True

    #     if self._is_out_of_bounds(robot_xy):
    #         rospy.logwarn("Robot hit wall or exited safe area!")
    #         reward -= 100.0
    #         terminated = True
    #         truncated = False
    #         self.cmd_pub.publish(Twist())
    #         rospy.sleep(0.05)

    #     # Build info dict
    #     info = {
    #         "phase":     self.phase,
    #         "dist_rb":   float(np.linalg.norm(vec[2:4] - robot_xy)),
    #         "dist_bt":   float(np.linalg.norm(vec[2:4] - vec[4:6])),
    #         "wall_crash": self._is_out_of_bounds(robot_xy),
    #     }

    #     # —— Auto-reset logic starts here ——
    #     # Guarantee reward is a native Python float
    #     obs, reward, terminated, truncated, info = (
    #         obs,
    #         float(reward),
    #         terminated,
    #         truncated,
    #         info,
    #     )

    #     if terminated or truncated:
    #         # Call your own reset() and merge info if desired
    #         obs, info = self.reset()


    #     # —— End auto-reset logic ——

    #     return obs, reward, terminated, truncated, info

    def step(self, action):
        # ——— 1) Execute action ———
        twist = Twist()
        if action == 0:
            twist.linear.x = 1.0
        elif action == 1:
            twist.angular.z = 0.5
        elif action == 2:
            twist.angular.z = -0.5
        # action == 3 → stop (zero twist)
        self.cmd_pub.publish(twist)
        rospy.sleep(0.05)

        # ——— 2) Observe ———
        obs = self._get_obs()
        img = obs["image"]
        vec = obs["vector"]
        robot_xy = vec[0:2]

        # ——— 3) Initialize flags & compute base reward ———
        terminated = False
        truncated  = False
        reward, term_flag = self._compute_reward(obs)
        terminated = terminated or term_flag

        # ——— 4) Camera bonus ———
        if self.phase == 0:
            reward += self.camera_block_visibility_reward(img) * 5.0
        else:
            reward += self.centering_reward(img) * 5.0

        # ——— 5) Distance shaping ———
        block_xy = vec[2:4]
        goal_xy  = vec[4:6]
        dist_rb  = np.linalg.norm(block_xy - robot_xy)
        dist_bt  = np.linalg.norm(block_xy - goal_xy)
        epsilon  = 0.01

        reward -= 0.5  # step penalty
        if self.phase == 1 and self.prev_dist_bt is not None:
            delta = self.prev_dist_bt - dist_bt
            if   delta >  epsilon:
                reward += delta * 20.0
            elif delta < -epsilon:
                reward += delta * 30.0
            else:
                reward -= 10.0
        self.prev_dist_bt = dist_bt

        # ——— 6) Timeout logic ———
        elapsed = (rospy.Time.now() - self.episode_start_time).to_sec()
        if elapsed > 60:
            rospy.logwarn("Episode truncated: timeout reached!")
            reward    -= 5.0
            truncated  = True
            terminated = False
            self.cmd_pub.publish(Twist())
            rospy.sleep(0.05)

        # ——— 7) Crash logic ———
        if self._is_out_of_bounds(robot_xy):
            rospy.logwarn("Robot hit wall or exited safe area!")
            reward    -= 10.0
            terminated = True
            truncated  = False
            self.cmd_pub.publish(Twist())
            rospy.sleep(0.05)

        # ——— 8) Build info dict ———
        info = {
            "phase":      self.phase,
            "dist_rb":    float(dist_rb),
            "dist_bt":    float(dist_bt),
            "wall_crash": self._is_out_of_bounds(robot_xy),
        }

        # ——— 9) Auto-reset logic ———
        # ensure reward is a native Python float
        obs, reward, terminated, truncated, info = (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )

        if terminated or truncated:
            # call reset() and merge episode-end info
            new_obs, new_info = self.reset()
            new_info.update(info)
            return new_obs, reward, terminated, truncated, new_info

        # ——— 10) Normal return ———
        return obs, reward, terminated, truncated, info




    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) 
        self._set_model_pose("phantomx", 0.0, 0.0, 0.25)
        x_block_pos = np.random.uniform(-1.2, 1.2)
        y_block_pos = np.random.uniform(-1.2, 1.2)

        # x_block_pos = 1.5
        # y_block_pos = 1.5
        self._set_model_pose("push_block", x_block_pos, y_block_pos, 0.2)
        
        self.prev_dist_bt =None

        rospy.sleep(0.05)
        # Re‐seed if you like:
        if seed is not None:
            np.random.seed(seed)

        # Reset timer
        self.episode_start_time = rospy.Time.now()

        # Build return values
        obs = self._get_obs()
        info = {}
        self.phase = 0
        return obs, info

    # def reset(self, *, seed=None, options=None):
    #     super().reset(seed=seed) 
    #     # 1) Reset robot position using Gazebo services
    #     rospy.wait_for_service('/gazebo/set_model_state')
    #     try:
    #         from gazebo_msgs.srv import SetModelState
    #         from gazebo_msgs.msg import ModelState

    #         set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    #         # Create model state
    #         model_state = ModelState()
    #         model_state.model_name = 'phantomx'  # <<< Change this if your robot model name is different!
    #         model_state.pose.position.x = 0.0
    #         model_state.pose.position.y = 0.0
    #         model_state.pose.position.z = 0.2  # small lift to prevent ground collision
    #         model_state.pose.orientation.x = 0.0
    #         model_state.pose.orientation.y = 0.0
    #         model_state.pose.orientation.z = 0.0
    #         model_state.pose.orientation.w = 1.0  # facing straight

    #         model_state.twist.linear.x = 0.0
    #         model_state.twist.linear.y = 0.0
    #         model_state.twist.linear.z = 0.0
    #         model_state.twist.angular.x = 0.0
    #         model_state.twist.angular.y = 0.0
    #         model_state.twist.angular.z = 0.0

    #         set_state(model_state)
    #         rospy.sleep(0.2)  # give Gazebo some time

    #     except rospy.ServiceException as e:
    #         rospy.logerr(f"Service call failed: {e}")

    #     # 2) Reset environment variables
    #     self.episode_start_time = rospy.Time.now()
    #     self.prev_dist_bt = None

    #     # 3) Get initial observation
    #     obs = self._get_obs()

    #     return obs, {}



    def _get_obs(self):
        # Vector features
        base = np.zeros(6, dtype=np.float32)
        yaw = 0.0
        yaw_error_block = 0.0
        yaw_error_goal = 0.0

        if self.robot_pose and self.block_pose:
            robot_xy = np.array([
                self.robot_pose.position.x,
                self.robot_pose.position.y
            ], dtype=np.float32)
            block_xy = np.array([
                self.block_pose.position.x,
                self.block_pose.position.y
            ], dtype=np.float32)
            goal_xy = np.array(self.target_pos, dtype=np.float32)

            base = np.concatenate([robot_xy, block_xy, goal_xy])

            # robot yaw
            q = self.robot_pose.orientation
            yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

            # angle to block
            vec_block = block_xy - robot_xy
            angle_block = math.atan2(vec_block[1], vec_block[0])
            yaw_error_block = (angle_block - yaw + np.pi) % (2 * np.pi) - np.pi

            # angle to goal
            vec_goal = goal_xy - block_xy
            angle_goal = math.atan2(vec_goal[1], vec_goal[0])
            yaw_error_goal = (angle_goal - yaw + np.pi) % (2 * np.pi) - np.pi

        # IMU data
        if self.imu_data:
            la = self.imu_data.linear_acceleration
            av = self.imu_data.angular_velocity
            imu_arr = np.array([la.x, la.y, la.z, av.x, av.y, av.z], dtype=np.float32)
        else:
            imu_arr = np.zeros(6, dtype=np.float32)

        # Final vector observation
        vector_obs = np.concatenate([
            base,
            np.array([yaw, yaw_error_block, yaw_error_goal], dtype=np.float32),
            imu_arr
        ])

        # Camera image observation
        if self.last_image is not None:
            img = cv2.resize(self.last_image, (64, 64))
            img = img / 255.0
            img = img.transpose(2, 0, 1)  # (C, H, W)
            img = img.astype(np.float32)
        else:
            img = np.zeros((3, 64, 64), dtype=np.float32)

        return {
            "vector": vector_obs,
            "image": img
        }

    def camera_block_visibility_reward(self, img):
        img = img.transpose(1, 2, 0) 
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Blue color range for block (adjust if needed)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_area = np.sum(mask) / 255  # count blue pixels

        visibility_score = blue_area / (64 * 64)
        return visibility_score  # 0 to 1

    def centering_reward(self, img):
        img = img.transpose(1, 2, 0)  
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            dist_to_center = np.linalg.norm(np.array([cx, cy]) - np.array([32, 32]))
            max_dist = np.linalg.norm(np.array([32, 32]))

            centering_score = 1.0 - (dist_to_center / max_dist)
            return centering_score
        else:
            return 0.0
        
    def _compute_reward(self, obs):
        """
        Returns (reward, done) based on:
        • Phase 0: find block
        • Phase 1: push block to target
        • Camera bonuses in each phase
        • Three-way delta shaping (closer/farther/no-move)
        • Small time penalty each step
        """
        vec = obs["vector"]
        img       = obs["image"]
        robot_xy  = vec[0:2]
        block_xy  = vec[2:4]
        goal_xy   = vec[4:6]

        dist_rb = np.linalg.norm(block_xy - robot_xy)
        dist_bt = np.linalg.norm(block_xy - goal_xy)

        reward     = 0.0
        terminated = False
        epsilon    = 0.01   # threshold for “no significant move”

        # --- PHASE 0: FIND THE BLOCK ---
        if self.phase == 0:
            # Base proximity reward + visibility bonus
            reward += 10.0 - dist_rb
            reward += self.camera_block_visibility_reward(img) * 5.0

            # Delta shaping on dist_rb
            if self.prev_dist_rb is not None:
                delta_rb = self.prev_dist_rb - dist_rb
                if   delta_rb >  epsilon:
                    reward += delta_rb * 10.0   # moved closer
                elif delta_rb < -epsilon:
                    reward += delta_rb * 15.0   # moved away
                else:
                    reward -= 0.5               # no movement
            self.prev_dist_rb = dist_rb

            # Transition to phase 1?
            if dist_rb <= self.find_threshold:
                reward     += 100.0
                self.phase  = 1

        # --- PHASE 1: PUSH TO TARGET ---
        else:
            # Centering bonus
            reward += self.centering_reward(img) * 5.0

            # Delta shaping on dist_bt
            if self.prev_dist_bt is not None:
                delta_bt = self.prev_dist_bt - dist_bt
                if   delta_bt >  epsilon:
                    reward += delta_bt * 20.0   # pushed closer
                elif delta_bt < -epsilon:
                    reward += delta_bt * 30.0   # pushed away
                else:
                    reward -= 1.0               # no movement
            self.prev_dist_bt = dist_bt

            # Success?
            if dist_bt <= self.goal_threshold:
                reward     += 100.0
                terminated = True

        # --- SMALL TIME PENALTY TO ENCOURAGE SPEED ---
        reward -= 0.1

        return float(reward), terminated




    def _set_model_pose(self, name, x, y, z):
        pose_msg = ModelState()
        pose_msg.model_name = name
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.w = 1.0

        pose_msg.twist.linear.x = 0.0
        pose_msg.twist.linear.y = 0.0
        pose_msg.twist.linear.z = 0.0
        pose_msg.twist.angular.x = 0.0
        pose_msg.twist.angular.y = 0.0
        pose_msg.twist.angular.z = 0.0
        pose_msg.reference_frame = "world"
        try:
            self.set_model_srv(pose_msg)
        except rospy.ServiceException as e:
            rospy.logwarn(f"Failed to set pose for {name}: {e}")
