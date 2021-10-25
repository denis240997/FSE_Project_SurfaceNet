FROM ubuntu:16.04

COPY ./app /app
COPY ./cudnn-8.0-linux-x64-v5.1.tgz /

ENV DEBIAN_FRONTEND "noninteractive"

# nvidia-375 driver
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN apt-get update && apt-get install -y nvidia-375


# cuda download
RUN wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
RUN dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
RUN rm cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
RUN apt-key add /var/cuda-repo-8-0-local-ga2/7fa2af80.pub
RUN apt-get update && apt-get install -y cuda

ENV PATH $PATH:/usr/local/cuda/bin
ENV CUDA_ROOT /usr/local/cuda


# cudnn download
RUN tar xvf cudnn-8.0-linux-x64-v5.1.tgz
RUN rm cudnn-8.0-linux-x64-v5.1.tgz
WORKDIR /cuda
RUN cp */*.h /usr/local/cuda/include/    # cp include/cudnn.h /usr/local/cuda/include
RUN cp */libcudnn* /usr/local/cuda/lib64/    # cp lib64/libcudnn* /usr/local/cuda/lib64
RUN chmod a+r /usr/local/cuda/lib64/libcudnn*
WORKDIR /app

ENV CUDNN_ROOT $CUDA_ROOT
ENV LD_LIBRARY_PATH $CUDNN_ROOT/lib64
ENV LIBRARY_PATH $CUDNN_ROOT/lib64
ENV CPATH $CUDNN_ROOT/include


ENTRYPOINT ["bash"]

