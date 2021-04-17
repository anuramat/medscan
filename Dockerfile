# base
FROM ubuntu:18.04
# install curl
RUN apt update &&  apt install -y curl
# install miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
# copy application files
COPY app /app
WORKDIR /app
# create environment
RUN conda update -n base -c defaults conda
RUN conda env create --file=frozen_env.yml
# install dependency for opencv
RUN apt install -y libgl1-mesa-glx
# "activate" environment 
SHELL ["conda", "run", "-n", "medscan", "/bin/bash", "-c"]
# open port
EXPOSE 5000
# start application
CMD uvicorn app:app --host 0.0.0.0 --port 5000 
