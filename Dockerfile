FROM nvcr.io/nvidia/tensorflow:18.06-py3
COPY requirements.txt requirements.txt
RUN apt update && apt install --yes software-properties-common python-software-properties \
    && add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt update \
    && apt install --yes python3.6-dev openjdk-8-jdk  \
    && curl https://bootstrap.pypa.io/get-pip.py | python3.6 \
    && python3.6 -m pip install --upgrade pip \
    && python3.6 -m pip install numpy \
    && python3.6 -m pip install -r requirements.txt \
    && rm /usr/bin/python \
    && ln -s /usr/bin/python3.6 /usr/bin/python \
    && export LD_LBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64