FROM python:3.10-slim
LABEL authors="tonychen"

WORKDIR /app

# Install git when pulling the source code from github. Not required for running inside the project source code.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container at /app
COPY requirements.txt /app/

# RUN #git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Define environment variable
# ENV NAME World

COPY . /app

EXPOSE 8510

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Camel.py", "--server.port=8510", "--server.address=0.0.0.0"]