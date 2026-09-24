FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

COPY deploy/render/exact_inspect_prepared_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY exact_inspect /app/exact_inspect

# Mount a verified prepared package at /data/package and, for local_app, a
# separate writable bundle library. Preparation and frontend builds are offline.
EXPOSE 10000
ENTRYPOINT ["python", "-m", "exact_inspect.cli", "serve"]
CMD ["--package", "/data/package/package.json", "--profile", "public_demo", "--host", "0.0.0.0", "--port", "10000"]
