FROM python:3.11-slim AS base
RUN apt-get update && apt-get install --no-install-recommends --yes \
    build-essential \
    git \
    vim \
    wget \
    procps \
    nano \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clone and install leds and its dependencies into the venv. A full clone
# gives setuptools_scm the git history it needs to derive the version. Override
# LEDS_REF to build a specific tag/branch/commit.
ARG LEDS_REPO=https://github.com/legend-exp/leds.git
ARG LEDS_REF=main
RUN git clone "${LEDS_REPO}" /src \
    && git -C /src checkout "${LEDS_REF}" \
    && uv pip install /src

# The array view renders through matplotlib (Agg, head-less) and Bokeh writes
# caches at runtime. Point caches/HOME at writable /tmp so the container works
# under an arbitrary non-root UID, which Spin assigns via runAsUser.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    PORT=5006

# Runtime configuration is supplied by the Spin service, not baked in:
#   LEDS_BASE_PATH        production-cycle directory (a mounted NERSC global
#                         filesystem, e.g. CFS); `leds serve` falls back to it.
#   BOKEH_ALLOW_WS_ORIGIN comma-separated public host[:port] allowed to open a
#                         websocket (the Spin hostname). Bokeh reads this env
#                         var directly; required when proxied behind Spin's LB.
EXPOSE 5006
CMD ["sh", "-c", "leds serve --address 0.0.0.0 --port ${PORT}"]
