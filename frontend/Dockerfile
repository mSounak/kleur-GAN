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

COPY . /frontend/

EXPOSE 8501

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

CMD ["streamlit", "run", "frontend/client.py"]