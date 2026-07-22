FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY pyproject.toml .

# Install the package
RUN pip install --no-cache-dir -e .

# Copy .env file (will be overridden by docker-compose env_file)
# .env is in .gitignore, so it won't be in the build context unless explicitly copied
# Use docker-compose env_file or environment variables instead

EXPOSE 8000

CMD ["python", "-m", "user_input_normalization.server"]
