# ********************************** STAGE 2 **********************************************
# later we will use the remote ones
FROM liehe/mlbench:mlbench-worker-dev

# -------------------- Debug --------------------
RUN apt-get update && apt-get install -y vim net-tools

# The reference implementation and user defined implementations are placed here.
RUN mkdir /codes
ENV PYTHONPATH=/codes