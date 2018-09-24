FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 as mlbench-worker-base
# TODO: reduce size and complexity of image.

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    libc-dev \
    musl-dev \
    openssh-server \
    g++ \
    git \
    curl \
    sudo

# -------------------- SSH --------------------
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

ARG SSH_USER=root
ENV SSH_USER=$SSH_USER
RUN mkdir -p /ssh-key/$SSH_USER && chown -R $SSH_USER:$SSH_USER /ssh-key/$SSH_USER
RUN mkdir -p /.sshd/host_keys && \
  chown -R $SSH_USER:$SSH_USER /.sshd/host_keys && chmod 700 /.sshd/host_keys
RUN mkdir -p /.sshd/user_keys/$SSH_USER && \
  chown -R $SSH_USER:$SSH_USER /.sshd/user_keys/$SSH_USER && chmod 700 /.sshd/user_keys/$SSH_USER
VOLUME /ssh-key/$SSH_USER

# -----–––---------------------- Cuda Dependency --------------------
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        --allow-change-held-packages \
         libnccl2=2.0.5-3+cuda9.0 \
         libnccl-dev=2.0.5-3+cuda9.0 &&\
     rm -rf /var/lib/apt/lists/*

# -------------------- Conda environment --------------------
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     sh ~/miniconda.sh -b -p /conda && rm ~/miniconda.sh
ENV PATH /conda/bin:$PATH
ENV LD_LIBRARY_PATH /conda/lib:$LD_LIBRARY_PATH

# TODO: Source code in Channel Anaconda can be outdated, switch to conda-forge if posible.
RUN conda install -y -c anaconda numpy pyyaml scipy mkl setuptools cmake cffi mkl-include typing \
    && conda install -y -c mingfeima mkldnn \
    && conda install -y -c soumith magma-cuda90 \
    && conda install -y -c conda-forge python-lmdb opencv numpy \
    && conda clean --all -y

# -------------------- Open MPI --------------------
RUN mkdir /.openmpi/
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz\
    && gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=/.openmpi/ --with-cuda\
    && make all install \
    && rm /openmpi-3.0.0.tar.gz \
    && rm -rf /openmpi-3.0.0 \
    && apt-get remove -y wget

ENV PATH /.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH /.openmpi/lib:$LD_LIBRARY_PATH

RUN mv /.openmpi/bin/mpirun /.openmpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /.openmpi/bin/mpirun && \
    echo "/.openmpi/bin/mpirun.real" '--allow-run-as-root "$@"' >> /.openmpi/bin/mpirun && \
    chmod a+x /.openmpi/bin/mpirun

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
RUN echo "hwloc_base_binding_policy = none" >> /.openmpi/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /.openmpi/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /.openmpi/etc/openmpi-mca-params.conf

# configure the path.
RUN echo export 'PATH=$HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin:$HOME/.openmpi/bin:$PATH' >> ~/.bashrc
RUN echo export 'LD_LIBRARY_PATH=$HOME/.openmpi/lib:$LD_LIBRARY_PATH' >> ~/.bashrc


# -------------------- PyTorch --------------------
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Install basic dependencies
# RUN git clone --recursive https://github.com/pytorch/pytorch && cd pytorch && python setup.py install
RUN git clone --recursive https://github.com/pytorch/pytorch && \
    cd pytorch && \
    git submodule update --init && \
    TORCH_CUDA_ARCH_LIST="3.5 3.7 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v . && \
    rm -rf /pytorch



RUN git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .

# RUN pip install -U git+https://github.com/ppwwyyxx/tensorpack.git

# RUN conda install -y -c anaconda msgpack
# RUN conda install -y -c anaconda msgpack msgpack-numpy pyzmq pillow
# RUN conda install -y -c conda-forge tqdm 
# # RUN conda install -y -c pchrapka zmq
# # RUN conda install -c omnia termcolor

# -------------------- patch --------------------
# libGL.so.1 might be lost when nvidia driver is installed
# sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglfw3-dev libgles2-mesa-dev
RUN apt-get install -y libgl1-mesa-glx


# -------------------- Others --------------------
RUN echo "orte_keep_fqdn_hostnames=t" >> /.openmpi/etc/openmpi-mca-params.conf

ADD ./compose/worker/entrypoint.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/entrypoint.sh

# Copy your application code to the container (make sure you create a .dockerignore file if any large files or directories should be excluded)
RUN mkdir /app/
WORKDIR /app/

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/sbin/sshd","-eD", "-f", "/.sshd/user_keys/root/sshd_config"]
