FROM python:3.8.12

RUN pip install tensorflow==2.6 \
    pip install keras==2.6 \
    pip install tensorflow-estimator==2.6 \
    pip install ray[rllib]==1.8 \
    pip install gym-retro \
    && pip install opencv-python

RUN apt-get update && apt-get install -y curl 

RUN apt-get -y install ffmpeg

RUN mkdir /tmp/rom \
    && curl -L -o /tmp/rom/megaman2.zip https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed/Nintendo%20-%20Nintendo%20Entertainment%20System.zip/Mega%20Man%202%20%28USA%29.zip \
    && curl -L -o /tmp/rom/supermariokart.zip https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed/Nintendo%20-%20Super%20Nintendo%20Entertainment%20System.zip/Super%20Mario%20Kart%20%28USA%29.zip 

COPY ./environments /environments

RUN unzip /tmp/rom/megaman2.zip -d /tmp/rom/ \
    && mv /tmp/rom/*.nes /environments/MegaMan2-Nes/rom.nes \
    && unzip /tmp/rom/supermariokart.zip -d /tmp/rom/ \
    && mv /tmp/rom/*.sfc /environments/SuperMarioKart-Snes/rom.sfc \
    && rm -rf /tmp/rom

WORKDIR /turing-retro

ENTRYPOINT ["/bin/bash", "-c", "cp -r -n /environments /turing-retro/ && /bin/bash"]
