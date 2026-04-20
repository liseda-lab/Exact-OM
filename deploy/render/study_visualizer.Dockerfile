FROM node:20-bookworm-slim AS frontend-build

WORKDIR /frontend

COPY explanations_visualizer/package*.json /frontend/

RUN --mount=type=cache,target=/root/.npm \
    npm config set fetch-retries 5 && \
    npm config set fetch-retry-factor 2 && \
    npm config set fetch-retry-mintimeout 10000 && \
    npm config set fetch-retry-maxtimeout 120000 && \
    npm config set prefer-offline true && \
    if [ -f package-lock.json ]; then \
      npm ci --no-audit --no-fund; \
    else \
      npm install --no-audit --no-fund; \
    fi

COPY explanations_visualizer/next-env.d.ts /frontend/
COPY explanations_visualizer/next.config.ts /frontend/
COPY explanations_visualizer/postcss.config.mjs /frontend/
COPY explanations_visualizer/tsconfig.json /frontend/
COPY explanations_visualizer/src /frontend/src

RUN \
    npm run build


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app

WORKDIR /app

COPY deploy/render/study_visualizer_requirements.txt /app/deploy/render/study_visualizer_requirements.txt
RUN python -m venv /opt/venv && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r /app/deploy/render/study_visualizer_requirements.txt

COPY deploy/render /app/deploy/render
COPY study_visualizer_runtime /app/study_visualizer_runtime
COPY --from=frontend-build /frontend/out /app/explanations_visualizer/out

RUN chmod +x /app/deploy/render/start_study_visualizer.sh

ENV EXACT_STUDY_ENABLE_ONTOLOGY_INFO=true \
    EXACT_STUDY_HOST=0.0.0.0 \
    EXACT_STUDY_LOG_LEVEL=INFO

EXPOSE 10000

CMD ["/app/deploy/render/start_study_visualizer.sh"]
