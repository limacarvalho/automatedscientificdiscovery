FROM ubuntu:22.04

LABEL project='Automated Scientific Discovery (ASD)'
LABEL container.version='1.3'
LABEL container.creation.date='23Oct2023'

COPY . /app

RUN chmod +x /app/*.sh
RUN bash -c '/app/build.sh'

EXPOSE 80

CMD bash -c '/app/init.sh'

