# Use the official Python slim image
FROM python:3.12.5-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of your current directory to /app in the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port (5000)
EXPOSE 5000

# Command to run your Flask app
CMD ["python", "app.py"]
