# Base image
FROM arm64v8/ros:noetic

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt upgrade -y
RUN apt install ros-noetic-desktop-full

# Update & install system tools, ROS packages, and Gazebo dependencies
RUN apt-get install -y \
    python3-pip \
    python3-tk \
    git \
    wget \
    curl \
    iputils-ping \
    lsb-release \
    sudo \
    vim \
    nano \
    unzip \
    terminator \
    # Core ROS packages
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
    ros-noetic-rosparam \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    stable-baselines3==2.4.1 \
    gym==0.21.0 \
    gymnasium==1.0.0 \
    numpy==1.24.4\
    matplotlib \
    opencv-python \
    pandas \
    PyYAML

# Create catkin workspace
# RUN apt-get install terminator -y
ENV PHANTOM_WS=/root/phantom_ws
RUN mkdir -p ${PHANTOM_WS}/src

# Set up ROS environment on shell startup
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /root/phantom_ws/devel/setup.bash" >> ~/.bashrc

# Default working directory
WORKDIR ${PHANTOM_WS}

# Copy source archive into workspace root
COPY src.zip ${PHANTOM_WS}/

# Unzip into workspace root and remove zip
RUN unzip ${PHANTOM_WS}/src.zip -d ${PHANTOM_WS}/ && rm ${PHANTOM_WS}/src.zip

# # Navigate into the src directory and clone hexapod_rl
# WORKDIR ${PHANTOM_WS}/src
# RUN git clone https://github.com/jonathandeepak/hexapod_rl.git .

# Return to the workspace root
WORKDIR ${PHANTOM_WS}

# # Clone PhantomX packages
# RUN git clone https://github.com/HumaRobotics/phantomx_gazebo.git ${PHANTOM_WS}/src/phantomx_gazebo && \
#     git clone https://github.com/HumaRobotics/phantomx_description.git ${PHANTOM_WS}/src/phantomx_description && \
#     git clone https://github.com/HumaRobotics/phantomx_control.git ${PHANTOM_WS}/src/phantomx_control && \
#     git clone https://github.com/tu-darmstadt-ros-pkg/hector_gazebo ${PHANTOM_WS}/src/hector_gazebo


# Install ROS package dependencies using rosdep
RUN apt-get update && \
    apt-get install -y python3-rosdep && \
    rosdep update

RUN rosdep install --from-paths src --ignore-src -r -y

# Build workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Source workspace on container start
RUN echo "source ${PHANTOM_WS}/devel/setup.bash" >> ~/.bashrc

# Enable Gazebo GUI from container
ENV DISPLAY=host.docker.internal:0
ENV QT_X11_NO_MITSHM=1

# Install X11 apps (optional)
RUN apt-get update && apt-get install -y x11-apps && rm -rf /var/lib/apt/lists/*

# Expose ROS master port
EXPOSE 11311

# Start a bash shell by default
CMD ["/bin/bash"]
