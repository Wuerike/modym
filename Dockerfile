FROM ubuntu:18.04

# Jupyter port
EXPOSE 8888

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y \
    software-properties-common \
    gfortran-7 \
    make \
    wget 
    
RUN rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Conda Install
RUN conda config --add channels conda-forge
RUN conda install pyfmi
RUN conda install cython
RUN conda install matplotlib
RUN conda install pyglet
RUN conda install jupyter

# Pip Install
RUN pip install gymnasium
RUN pip install stable-baselines3
RUN pip install pygame

# Export display
RUN apt-get update && apt-get install xvfb python-opengl -y
RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

WORKDIR /workspaces/modym
