FROM ubuntu
# TODO: reduce size and complexity of image.

RUN apt-get update && apt-get install -y \
    gcc \
    make \
    libc-dev \
    musl-dev \
    openssh-server

RUN apt-get update && apt-get install -y \
    g++

RUN apt-get update && apt-get install -y \
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

# -------------------- Conda environment --------------------
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     sh ~/miniconda.sh -b -p /conda
ENV PATH /conda/bin:$PATH
ENV LD_LIBRARY_PATH /conda/lib:$LD_LIBRARY_PATH

RUN conda install -y -c anaconda numpy pyyaml scipy mkl setuptools cmake cffi mkl-include typing
RUN conda install -y -c mingfeima mkldnn
# RUN conda install -y -c conda-forge openmpi

# -------------------- Open MPI --------------------
RUN mkdir /.openmpi/
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
RUN gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=/.openmpi/ \
    && make all install

ENV PATH /.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH /.openmpi/lib

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
RUN git clone --recursive https://github.com/pytorch/pytorch && cd pytorch && python setup.py install

# -------------------- Debug --------------------
RUN apt-get install -y vim net-tools



# -------------------- Others --------------------
RUN echo "orte_keep_fqdn_hostnames=t" >> /.openmpi/etc/openmpi-mca-params.conf

ADD ./compose/worker/entrypoint.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/entrypoint.sh

# Copy your application code to the container (make sure you create a .dockerignore file if any large files or directories should be excluded)
RUN mkdir /app/
WORKDIR /app/

ADD ./mlbench/worker/ /app/

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/sbin/sshd","-eD", "-f", "/.sshd/user_keys/root/sshd_config"]