FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# # Copy the current directory contents into the container at /app
ADD . /app

# ADD requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app/app.py"]
