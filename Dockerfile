FROM rayproject/ray-ml

RUN pip install gym-retro \
    && pip install pandas

RUN sudo apt-get update && sudo apt-get install -y curl

RUN mkdir /tmp/rom \
    && curl -L -o /tmp/rom/megaman2.zip https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed/Nintendo%20-%20Nintendo%20Entertainment%20System.zip/Mega%20Man%202%20%28USA%29.zip \
    && python3 -m retro.import /tmp/rom \
    && rm -rf /tmp/rom

COPY ./environments ./environments