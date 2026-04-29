FROM ultralytics/ultralytics:latest

WORKDIR /work

COPY src ./src
COPY configs ./configs
COPY vehicle.yaml ./vehicle.yaml