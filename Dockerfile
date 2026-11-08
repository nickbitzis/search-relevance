# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv, app code, and the saved evaluation report.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 searchrel
WORKDIR /app
COPY --from=build --chown=searchrel:searchrel /app /app
COPY --chown=searchrel:searchrel reports ./reports
# The service may write a fresh report here, so the directory must belong to the runtime user.
RUN mkdir -p /app/reports && chown -R searchrel:searchrel /app/reports
USER searchrel
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8846 SEARCHREL_REPORTS=/app/reports
EXPOSE 8846
# Startup generates the catalog and query log, builds the index, and trains the models: about 60 to 120 seconds
# at full size on two cores. Set SEARCHREL_PRODUCTS, SEARCHREL_SEARCHES, SEARCHREL_JUDGED to go smaller.
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8846/healthz', timeout=2)"
CMD ["searchrel", "serve"]
