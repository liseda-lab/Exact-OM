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
COPY deploy/render/exact_study_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Keep the complete small service module graph in sync with the prepared reader.
# Native parsers/providers remain offline-only; no preparation runs in this image.
COPY exact_inspect /app/exact_inspect
# Only /participate/ and /admin/ are served from this export; exploration pages are 404.
COPY --from=frontend-build /frontend/out /app/exact_inspect/static
# Populate this read-only prepared package before publication; no private keys here.
RUN mkdir -p /app/study-assets
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn exact_inspect.study.api:app_from_env --factory --host 0.0.0.0 --port ${PORT:-10000} --no-access-log"]
