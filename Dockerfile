# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source files
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Run Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
