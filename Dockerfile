FROM continuumio/miniconda3

LABEL maintainer="Pragathieshwaran"

RUN git clone https://github.com/Pragathiesh/mle-training.git

COPY deploy/conda/linux_cpu_py39.yml env.yml

RUN conda env create -n housing -f env.yml
RUN pip install -U pytest


RUN cd mle-training \
    && conda run -n housing python3 setup.py install\
    && cd src/house_price\
    && conda run -n housing python3 ingest_data.py\
    && conda run -n housing python3 train.py\
    && conda run -n housing python3 score.py

RUN cd mle-training\
    && cd tests/unit_tests\
    && conda run -n housing python3 ingest_data_test.py\
    && conda run -n housing python3 train_test.py\
    && conda run -n housing python3 score_test.py
    
RUN cd mle-training\
    && conda run -n housing pytest tests/functional_tests