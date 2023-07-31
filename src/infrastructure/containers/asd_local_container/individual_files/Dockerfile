FROM tensorflow/tensorflow:2.13.0
LABEL "com.asd.project"="Automated Scientific Discovery"
LABEL version="1.1"
LABEL description="The Automated Scientific Discovery project is a Python module/app that automatically discovers hidden relationships in the measurement data."
COPY . /opt/asd/
RUN chmod +x /opt/asd/setup_init.sh && /opt/asd/setup_init.sh
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib/:/usr/local/cuda-11.0/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
ENV PYTHONPATH="/opt/asd/python-asd/src/asd:/opt/asd/python-asd/src/asd/complexity:/opt/asd/python-asd/src/asd/complexity/dim_reduce:/opt/asd/python-asd/src/asd/predictability:/opt/asd/python-asd/src/asd/relevance:/opt/asd/python-asd/src/asd/relevance/ml:/opt/asd/python-asd/src/asd/relevance/utils"
EXPOSE 80/tcp
RUN chmod +x /opt/asd/asd_exec.sh
CMD /opt/asd/asd_exec.sh