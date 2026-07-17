FROM ultralytics/ultralytics:latest

WORKDIR /work

RUN pip install flask
RUN pip install sahi
COPY src ./src
COPY configs ./configs
COPY vehicle.yaml ./vehicle.yaml
COPY services ./services
COPY routes ./routes
COPY deploy ./deploy
COPY app.py ./app.py
COPY templates ./templates

EXPOSE 5000

CMD ["python", "app.py"]

