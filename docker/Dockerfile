FROM nvcr.io/nvidia/pytorch:21.09-py3
ENV DEBIAN_FRONTEND=noninteractive 

# dependencies for gym
#
RUN apt-get -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update \
    && apt-get install -y --allow-unauthenticated --no-install-recommends \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    zip \
    unzip \
    make \
    gcc-8 \
    g++-8 \
    vulkan-utils \
    mesa-vulkan-drivers \
    pigz \
    git \
    libegl1 \
    git-lfs \
    ffmpeg \
    sudo

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt/isaacgym

RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

RUN useradd --create-home gymuser -p "$(openssl passwd -1 ubuntu)"
RUN usermod -aG sudo gymuser
# Ensure sudo group users are not 
# asked for a password when using 
# sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> \
    /etc/sudoers
USER gymuser

# copy gym repo to docker
COPY --chown=gymuser docker/python /opt/isaacgym/python
# install gym modules
ENV PATH="/home/gymuser/.local/bin:$PATH"
RUN cd /opt/isaacgym/python && pip install -q -e .

WORKDIR /home/gymuser


# RUN git clone https://github.com/ZiwenZhuang/parkour/ /home/gymuser/parkour

# RUN cd parkour/rsl_rl && pip install -e .

# RUN cd parkour/legged_gym && pip install -e .

# RUN pip install "protobuf==3.20.*"

RUN sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
RUN sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

COPY --chown=gymuser:gymuser extreme-parkour /home/gymuser/extreme-parkour
RUN chmod -R u+rwX /home/gymuser/extreme-parkour

RUN cd extreme-parkour/rsl_rl && pip install -e .
RUN cd extreme-parkour/legged_gym && pip install -e .
RUN pip install "numpy<1.24" pydelatin "wandb<0.19.11" tqdm opencv-python ipdb pyfqmr flask
RUN pip install --upgrade pydantic

USER root
COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 755 /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
