FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_MAX_THREADS=2

RUN sed -i \
      -e 's@http://deb.debian.org/debian-security@https://mirrors.cloud.tencent.com/debian-security@g' \
      -e 's@http://deb.debian.org/debian@https://mirrors.cloud.tencent.com/debian@g' \
      /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates tzdata curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.lock.txt ./
RUN sed '/^-e \.\//d; /^-e \.$/d' requirements.lock.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir --index-url https://mirrors.cloud.tencent.com/pypi/simple -r /tmp/requirements.txt
COPY ashare ./ashare
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8765
CMD ["python", "-m", "ashare", "serve", "--host", "0.0.0.0", "--port", "8765"]
