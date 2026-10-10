FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NUMBA_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    gcc \
    g++ \
    ffmpeg \
    libopenblas-dev \
    liblapack-dev \
    libgfortran5 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV LD_LIBRARY_PATH="/workspace/msa_platform/libs:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /workspace/msa_platform

COPY requirements.txt ./

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
       torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser . /workspace/msa_platform

# 这些大文件不进入 Git；构建前必须按 README 放到指定路径。
RUN test -f /workspace/msa_platform/lnln_deploy.onnx \
    && test -x /workspace/msa_platform/MSA_FET/exts/OpenFace/FeatureExtraction \
    && test -d /workspace/msa_platform/libs \
    && cd /workspace/msa_platform/libs \
    && for f in libopencv_*.so.4.1.0; do \
         [ -f "$f" ] && ln -sf "$f" "${f%.0}"; \
       done \
    && if [ -f libdc1394.so.22.2.1 ]; then ln -sf libdc1394.so.22.2.1 libdc1394.so.22; fi \
    && if [ -f libgdk-x11-2.0.so.0.2400.32 ]; then ln -sf libgdk-x11-2.0.so.0.2400.32 libgdk-x11-2.0.so.0; fi \
    && if [ -f libgtk-x11-2.0.so.0.2400.32 ]; then ln -sf libgtk-x11-2.0.so.0.2400.32 libgtk-x11-2.0.so.0; fi \
    && if [ -f libatk-1.0.so.0.23510.1 ]; then ln -sf libatk-1.0.so.0.23510.1 libatk-1.0.so.0; fi \
    && if [ -f libXcomposite.so.1.0.0 ]; then ln -sf libXcomposite.so.1.0.0 libXcomposite.so.1; fi \
    && if [ -f libXdamage.so.1.1.0 ]; then ln -sf libXdamage.so.1.1.0 libXdamage.so.1; fi \
    && mkdir -p /workspace/msa_platform/uploads /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /workspace/msa_platform /home/appuser

ENV HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface

USER appuser

EXPOSE 8000

CMD ["uvicorn", "start_engine_API:app", "--host", "0.0.0.0", "--port", "8000"]
