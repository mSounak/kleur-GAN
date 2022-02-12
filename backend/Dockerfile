FROM python:3.8-slim-buster AS compile-image
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.8-slim-buster AS build-image
COPY --from=compile-image /opt/venv /opt/venv

COPY src/utils.py /backend/
COPY src/generator_model.py /backend/
COPY src/app.py/ /backend/
COPY src/config.py/ /backend/
COPY models/checkpoints/gen.pth.tar /backend/models/checkpoints/

EXPOSE 8000

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

CMD ["python3", "backend/app.py"]