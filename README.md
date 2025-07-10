
---

## 🔧 Setup Instructions for running the Simulation in Docker🐳

Download **src.zip** and **dockerfile** from docker branch into the same directory

Build the docker image
```bash
docker build -t enpm690rlgroup2 .
```
Run the Container with GUI support inside WSL **(Preferred)** or Ubuntu Desktop

```bash
xhost +local:root
```
```bash

docker run --rm -it --net=host --gpus all -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix enpm690rlgroup2

```

Run the Container with GUI support using Docker desktop + Xming server setup (for windows)
```bash
docker run --rm -it --gpus all -e DISPLAY=host.docker.internal:0 enpm690rlgroup2
```


### Launch the simulation
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
