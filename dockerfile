FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install numpy==2.2.5
RUN pip install git+https://github.com/numba/numba.git@main
# Copy app code and models
COPY . .

ENV PORT=8081

CMD ["python", "app.py"]
