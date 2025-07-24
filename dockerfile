FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and models
COPY . .

ENV PORT=8081

CMD ["python", "app.py"]
