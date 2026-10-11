FROM python:3.12.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/opt/token-cache AML_PORT=8000 AML_HOST=0.0.0.0
WORKDIR /srv/aml
COPY requirements-cycle2.txt ./requirements-cycle2.txt
RUN python -m pip install --no-cache-dir --no-deps -r requirements-cycle2.txt \
    && python -m pip check
# The encoding asset is fetched at build time, never at server startup.
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
COPY app ./app
COPY deploy ./deploy
RUN python -m deploy.runtime \
    && groupadd --gid 10001 aml \
    && useradd --uid 10001 --gid 10001 --no-create-home aml \
    && mkdir -p /var/lib/aml \
    && chown 10001:10001 /var/lib/aml
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=6s --start-period=30s --retries=3 \
    CMD ["python", "-m", "deploy.healthcheck"]
CMD ["python", "-m", "deploy.start"]
