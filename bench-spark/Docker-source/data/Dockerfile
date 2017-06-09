FROM ubuntu:16.04

# Install packages
RUN BUILD_DEPS="wget bunzip2" \
    && apt-get update && apt-get install -y --no-install-recommends ${BUILD_DEPS} \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data \
    && wget --no-check-certificate -O /data/gisette_scale.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2 \
    && bunzip2 -d /data/gisette_scale.bz2 \
    && apt-get purge -y --auto-remove ${BUILD_DEPS}

VOLUME ["/data"]
