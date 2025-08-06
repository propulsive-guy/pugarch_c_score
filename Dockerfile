# ✅ Use Google's ML-ready base image (avoids Docker Hub issues)
FROM gcr.io/deeplearning-platform-release/base-cpu

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy all files from GitHub repo into the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Cloud Run uses
EXPOSE 8080

# Run app using Gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
