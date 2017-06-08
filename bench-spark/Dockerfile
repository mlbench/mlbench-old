FROM spark21

RUN apt-get update -y && apt-get install -y apt-transport-https ca-certificates wget

# installing scala
RUN apt-get remove -y scala-library scala
RUN wget www.scala-lang.org/files/archive/scala-2.11.7.deb
RUN dpkg -i scala-2.11.7.deb

# Check the scala version
RUN scala -version

RUN echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
RUN apt-get update -y 
RUN apt-get install -y --no-install-recommends sbt=0.13.8
RUN sbt sbtVersion
#RUN sbt -v

# Benchmark files
COPY ./benchmark /root/benchmark

WORKDIR /root/benchmark

#compile the program
RUN sbt assembly

#starting point




