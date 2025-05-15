
---

## 🔧 Setup Instructions

### 1. Clone and Build ROS Workspace

```bash
mkdir -p ~/phantom_ws/src
cd ~/phantom_ws/src

git clone https://github.com/jonathandeepak/hexapod_rl .
git clone https://github.com/HumaRobotics/phantomx_gazebo.git
git clone https://github.com/HumaRobotics/phantomx_description.git
git clone https://github.com/HumaRobotics/phantomx_control.git

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
Launch Terminator
```bash
terminator
```
**Vertical Split (side-by-side)** | `Ctrl` + `Shift` + `E` |

**Horizontal Split (top and bottom)** | `Ctrl` + `Shift` + `O` |

Terminal 1 – Launch Gazebo simulation:
```bash
source ~/phantom_ws/devel/setup.bash
roslaunch hexapod_rl phantomx_gazebo_combined.launch
```
Terminal 2 – Run DQN model inference script
```bash
source ~/phantom_ws/devel/setup.bash
rosrun hexapod_rl rl_inference.py
```
Terminal 3 – Run PPO model inference script (stop DQN inference script before running this)
```bash
source ~/phantom_ws/devel/setup.bash
rosrun hexapod_rl rl_ppo_inference.py
```

### 🐳 5. (Optional) Run the Simulation in Docker
Copy **src.zip** and **dockerfile** into the same directory

Build the docker image
```bash
docker build -t enpm690rlgroup2 .
```
Run the Container with GUI support inside WSL **(Preferred)** or Ubuntu Desktop

```bash
xhost +local:root

docker run --rm -it --net=host --gpus all -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix enpm690rlgroup2

```

Run the Container with GUI support using Docker desktop + Xming server setup (for windows)
```bash
docker run --rm -it --gpus all -e DISPLAY=host.docker.internal:0 enpm690rlgroup2
```

