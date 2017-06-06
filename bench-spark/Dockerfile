FROM cloudsuite/spark

# Benchmark files
RUN mkdir /root/benchmark
COPY build.sbt /root/benchmark
COPY ./src /root/benchmark
COPY ./libs /root/benchmark
COPY ./projects /root/benchmark

WORKDIR /root/benchmark

RUN sbt assembly


