FROM python:3.8-slim-buster AS compile-image
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r backend/requirements.txt

FROM python:3.8-slim-buster AS build-image
COPY --from=compile-image /opt/venv /opt/venv

COPY backend/src/utils.py /backend/
COPY backend/src/generator_model.py /backend/
COPY backend/src/app.py/ /backend/
COPY backend/src/config.py/ /backend/
COPY backend/src/res_unet.py /backend/
COPY backend/src/dataset.py /backend/
# COPY backend/models/checkpoints/gen.pth.tar /backend/models/checkpoints/


# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

CMD ["python3", "backend/app.py"]