# 1. Use the official lightweight Python 3.11 image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your app code and the trained model
COPY app.py .
COPY models/ ./models/

# 5. Expose the web port
EXPOSE 8000

# 6. Command to start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]