FROM ubuntu:24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/bin/

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt update && apt install -y curl && apt clean

COPY pyproject.toml .

RUN uv sync

RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . .

ENTRYPOINT ["uv", "run", "--no-sync", "main.py", "--mode", "api"]
