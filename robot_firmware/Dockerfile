FROM osrf/ros:humble-desktop

ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y nano git python3-pip ffmpeg sudo python3-tk libx11-dev python3-pyqt5


# pilllow requires below
RUN apt install -y libjpeg-dev zlib1g-dev


# COPY ./requirements.txt /tmp/requirements.txt
# RUN pip install -r /tmp/requirements.txt

RUN pip install pyrealsense2 matplotlib numpy opencv-python

ENV USERNAME=developer
# we use uid of 1000, seems to be default in ubuntu but change if different
RUN useradd -U --uid 1000 -ms /bin/bash $USERNAME \
    && echo "$USERNAME:$USERNAME" | chpasswd \
    && adduser $USERNAME sudo \
    && adduser $USERNAME dialout \
    && echo "$USERNAME ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USERNAME
# Commands below run as the developer user
USER $USERNAME
# When running a container start in the developer's home folder
WORKDIR /home/$USERNAME

COPY realsense.py .
COPY visualize_realsense.py .
ENV MPLBACKEND=tkagg

RUN mkdir -p ros_ws/src

RUN git clone https://github.com/unitreerobotics/unitree_ros2 ros_ws/src/unitree_ros2 && cd ros_ws/src/unitree_ros2 && git checkout 532ae526a35e8587ac3e81d657f126f959842aba
RUN sudo apt install -y ros-humble-rmw-cyclonedds-cpp ros-humble-rosidl-generator-dds-idl

RUN git clone https://github.com/ros2/rmw_cyclonedds -b humble ros_ws/src/rmw_cyclonedds
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x ros_ws/src/cyclonedds

SHELL ["/bin/bash", "-c"]

RUN cd ros_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select cyclonedds
RUN cd ros_ws && source /opt/ros/humble/setup.bash && colcon build
RUN sudo apt-get install -y net-tools


RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

RUN sudo apt-get install -y tmux

RUN pip install pybind11

# compiling crc check for our python version
# for the python cnc check
RUN cd /tmp && git clone https://github.com/yihuai-gao/unitree_sdk2.git && cd unitree_sdk2/python && mkdir build && cd build && cmake .. -DPYBIND_PATH=$(pip show pybind11 | grep Location | cut -d' ' -f2) && make

# for the low level examples that disable sports mode. Don't load env vars when you do this.
RUN git clone https://github.com/unitreerobotics/unitree_sdk2.git && cd unitree_sdk2 && git checkout 6b72f4e78fc9b0b0bf5fbb27596763f4551dfdb6 && mkdir build && cd build && cmake .. && make

RUN sudo apt-get install ros-humble-pcl-ros -y


RUN cd /tmp && git clone https://github.com/Livox-SDK/Livox-SDK2.git && cd ./Livox-SDK2/ && mkdir build && cd build && cmake .. && make -j1 && sudo make install


RUN cd ~/ros_ws/src && git clone https://github.com/Ericsii/FAST_LIO_ROS2 --recurse-submodules && git clone https://github.com/Ericsii/livox_ros_driver2.git && cd livox_ros_driver2 && git checkout 378d1c7c9d33ec44d0683aef885b6ed0cce9612c && cd ~/ros_ws && source /opt/ros/humble/setup.bash && colcon build

RUN pip install wandb

COPY setup.sh setup.sh
COPY go2_fastlio.yaml go2_fastlio.yaml 