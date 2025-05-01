
---

## 🔧 Setup Instructions

### 1. Clone and Build ROS Workspace

```bash
mkdir -p ~/phantom_ws/src
cd ~/phantom_ws/src
git clone https://github.com/your-username/hexapod-rl.git .
cd ..
catkin_make
source devel/setup.bash
```

### 2. Install ROS Dependencies
```bash
sudo apt update
cd ~/phantom_ws/
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```
or install manually via

```bash
sudo apt update && sudo apt install -y \
ros-noetic-roscpp \
ros-noetic-rospy \
ros-noetic-geometry-msgs \
ros-noetic-sensor-msgs \
ros-noetic-std-msgs \
ros-noetic-gazebo-ros \
ros-noetic-gazebo-plugins \
ros-noetic-controller-manager \
ros-noetic-effort-controllers \
ros-noetic-joint-state-controller \
ros-noetic-xacro \
ros-noetic-tf \
ros-noetic-rviz \
ros-noetic-roslaunch \
ros-noetic-rosparam
```
### 3. Install Python Dependencies
```bash
pip install stable-baselines3 gym numpy matplotlib opencv-python pandas PyYAML
```

### 4. Launch the simulation
Terminal 1 – Launch Gazebo simulation:
```bash
source ~/phantom_ws/devel/setup.bash
roslaunch hexapod_rl phantomx_gazebo_combined.launch
```
Terminal 2 – Run DQN model inference script
```bash
rosrun hexapod_rl hexapod_rl_training.py
```

