# ********************************** STAGE 2 **********************************************
# later we will use the remote ones
FROM liehe/mlbench:mlbench-worker-dev

# -------------------- Debug --------------------
RUN apt-get update && apt-get install -y vim net-tools

ADD ./mlbench/worker/ /app/

# The reference implementation and user defined implementations are placed here.
RUN mkdir /codes
ADD ./mlbench/refimpls/pytorch /codes

ENV PYTHONPATH /codes