FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install numpy==2.2.5
RUN pip install git+https://github.com/numba/numba.git@main
# Copy app code and models
COPY . .

ENV PORT=8081

CMD ["python", "app.py"]
