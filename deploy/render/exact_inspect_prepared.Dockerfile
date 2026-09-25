FROM node:20-bookworm-slim AS frontend-build
WORKDIR /frontend
COPY explanations_visualizer/package*.json /frontend/
RUN npm ci --no-audit --no-fund
COPY explanations_visualizer /frontend
# The static export is inert; the server decides per profile which pages exist.
RUN NEXT_TELEMETRY_DISABLED=1 npm run build

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

COPY deploy/render/exact_inspect_prepared_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY exact_inspect /app/exact_inspect
COPY --from=frontend-build /frontend/out /app/exact_inspect/static

# Mount a verified prepared package at /data/package and, for local_app, a
# separate writable bundle library. Preparation is offline; the image serves the
# exploration pages its profile allows (public_demo: compare and browse only).
EXPOSE 10000
ENTRYPOINT ["python", "-m", "exact_inspect.cli", "serve"]
CMD ["--package", "/data/package/package.json", "--profile", "public_demo", "--host", "0.0.0.0", "--port", "10000"]
