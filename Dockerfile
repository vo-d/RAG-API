# This file will create a docker image for the application

# Use the official Python image from Docker
FROM python

# Set the working directory in the container
WORKDIR /app

# Copy the requirement files in the working directory
COPY requirements.txt .

# Install all Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy local code to the container
COPY . /app

# Expose the port the app will be running on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]