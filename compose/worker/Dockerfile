# ********************************** STAGE 2 **********************************************
# later we will use the remote ones
FROM mlbench/mlbench_worker:mlbench-worker-base

# -------------------- Debug --------------------
RUN apt-get update && apt-get install -y vim net-tools iproute2

ADD ./mlbench/worker/ /app/

# The reference implementation and user defined implementations are placed here.
RUN mkdir /codes
ADD ./mlbench/refimpls/pytorch /codes

ENV PYTHONPATH /codes

ADD ./compose/worker/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# To find `libnvidia-ml.so` on google cloud.
# ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true
RUN pip install tensorpack