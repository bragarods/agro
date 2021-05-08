# set base image (host OS)
FROM python:3.9

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY / .

# set env var

ENV ARTIFACT_ROOT_ENV_VAR = "/ml_flow"

# command to run on container start
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]