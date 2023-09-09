FROM ubuntu:22.04

RUN apt update
RUN apt upgrade

# WORKDIR /Docker_shared

# RUN mkdir Representation\ Learning\ for\ Model-Predictive\ Exploration\ in\ Reinforcement\ Learning

WORKDIR /Docker_shared/Representation\ Learning\ for\ Model-Predictive\ Exploration\ in\ Reinforcement\ Learning

COPY ./ ./

# install python 3.8.16
ARG DEBIAN_FRONTEND=noninteractive
RUN apt install software-properties-common -y # required to use add-apt-repository command
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.8 -y
# change python3 to refer to 3.8 instead of 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 20

# install pip23.0.1
RUN apt install python3-pip -y
RUN apt install python3-distutils -y
RUN apt install python3-apt -y
RUN apt install python3.8-distutils -y
RUN python3 -m pip install pip==23.0.1 # Upgrade to a specific version

# install python3 requirements:
RUN apt install ffmpeg libsm6 libxext6 -y
RUN pip3 install numpy==1.24.2
RUN pip3 install torch==1.13.1
RUN pip3 install gym==0.21.0
RUN pip3 install gym-super-mario-bros==7.4.0
RUN pip3 install autorom==0.4.2
RUN pip3 install autorom-accept-rom-license==0.6.1
RUN pip3 install opencv-python==4.6.0.66
RUN pip3 install pillow==9.5.0
RUN pip3 install tensorboard==2.13.0
RUN pip3 install tensorboard-data-server==0.7.0
RUN pip3 install gym[atari,accept-rom-license]==0.21.0
