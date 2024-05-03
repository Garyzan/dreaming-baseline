FROM ubuntu:14.04

ENV PYTHONUNBUFFERED 1

# install conda and required build tools
RUN  apt-get update \
  && apt-get install -y wget libgl1-mesa-glx build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

WORKDIR /opt/app

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

# Environment setup
COPY --chown=user:user environment.yml /opt/app/
RUN /opt/conda/bin/conda env create --file environment.yml
RUN /opt/conda/bin/conda run -n e2fgvi python -m pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5/index.html

# collect algorithm files
COPY --chown=user:user core /opt/app/core
COPY --chown=user:user model /opt/app/model
COPY --chown=user:user release_model /opt/app/release_model
COPY --chown=user:user spynet_20210409-c6c1bd09.pth /home/user/.cache/torch/checkpoints/
COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user reconstruct.py /opt/app/

RUN /opt/conda/bin/conda run -n e2fgvi python reconstruct.py
RUN rm /opt/app/release_model/*.tmp*

ENTRYPOINT ["/opt/conda/bin/conda", "run", "-n", "e2fgvi", "python", "inference.py"]
