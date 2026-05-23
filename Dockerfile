# 1. 
FROM python:3.10-slim

# 2.
WORKDIR /app

# 3.
COPY requirements.txt .

# 4.
RUN pip install --no-cache-dir -r requirements.txt

# 5.
COPY . .

# 6.
EXPOSE 8501

# 7.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]