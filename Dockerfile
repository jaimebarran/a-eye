# FROM python:3.8
FROM python:3.6.5

# Create directory inside Docker's image
WORKDIR /a-eye_docker

# Copy requirements
COPY requirements.txt /a-eye_docker

# Install requirements
# no upgrade requirements
RUN pip install -r /a-eye_docker/requirements.txt 
# upgrade requirements & no cache directory
# RUN pip install --no-cache-dir --upgrade -r /a-eye_docker/requirements.txt

# Copy content from project to Docker's project directory
# TODO: do not copy data, only the trained model
# COPY ./a-eye_web /a-eye_docker/a-eye_web
COPY . /a-eye_docker

EXPOSE 5000

# Export environment variables. This will allow running $ flask run
ENV FLASK_APP=./a-eye_web/app.py
ENV FLASK_ENV=development
# ENV FLASK_DEBUG=1
# no cpu instructions warnings (test)
# ENV TF_CPP_MIN_LOG_LEVEL=2

# Execute command in terminal
# export FLASK_APP=a-eye_web/app.py
# export FLASK_ENV=development
# CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
# CMD ["python3", "-m", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]