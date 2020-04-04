FROM python:3.7

# Create work directory
RUN mkdir -p /opt/user-service
WORKDIR /opt/user-service

# Copy source
COPY ./user-service ./
COPY ./requirements.txt .

# Install python additional packages 
RUN pip3 install -r requirements.txt

EXPOSE 5000
CMD gunicorn -b :5000 main:app

