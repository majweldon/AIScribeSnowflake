# Use an official Python runtime as the base image
FROM python:3.9-slim

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the command to run when the container starts
CMD [ "python", "app.py" ]