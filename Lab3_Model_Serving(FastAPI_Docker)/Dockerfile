# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model and the application code into the container
COPY model.pkl .
COPY app.py .
COPY train_model.py .

# Make port 80 exposed to the outside world
EXPOSE 80

# Run the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]