FROM python:3.11-slim-bookworm

ARG TORCH_VERSION=2.9.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:99 \
    HOME=/home/sentinel \
    XDG_RUNTIME_DIR=/tmp/runtime-sentinel \
    SENTINEL_X_DATA_DIR=/home/sentinel/.local/share/sentinel-x \
    QT_X11_NO_MITSHM=1 \
    QT_QUICK_BACKEND=software \
    LIBGL_ALWAYS_SOFTWARE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates dbus-x11 fonts-dejavu-core libegl1 libgl1 libopengl0 \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxkbcommon-x11-0 \
    novnc supervisor thunar tini websockify x11-utils x11vnc \
    xfce4-panel xfce4-session xfce4-settings xfdesktop4 xfwm4 xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash sentinel

WORKDIR /opt/sentinel-x
COPY pyproject.toml requirements.txt README.md ./
COPY src/backend/ src/backend/
COPY docker/requirements.lock requirements.lock
RUN python -m pip install -c requirements.lock "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install -c requirements.lock . \
    && python -m pip check

RUN apt-get update && apt-get install -y --no-install-recommends nginx-light openssl \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/setup_environment.py scripts/setup_environment.py
COPY ["model weights/fe44637dfc4d49fe9cd402c4ce6cc21d.pt", "model weights/fe44637dfc4d49fe9cd402c4ce6cc21d.json", "model weights/"]
COPY results/final/checkpoint_map.json results/final/checkpoint_map.json
COPY docker/ /opt/container/
RUN chmod 755 /opt/container/*.sh \
    && mkdir -p /usr/local/share/applications \
    && cp /opt/container/*.desktop /usr/local/share/applications/ \
    && mkdir -p /home/sentinel/.local/share/sentinel-x \
    && chown -R sentinel:sentinel /home/sentinel

USER 1000:1000
EXPOSE 6080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "/opt/container/healthcheck.py"]
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/container/entrypoint.sh"]
