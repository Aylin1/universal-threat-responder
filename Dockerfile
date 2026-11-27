FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    DASHBOARD_CONTAINER=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt scikit-learn==1.9.0 xgboost==3.2.0 \
    && python -m pip check

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY app.py config.py ./
COPY adapters ./adapters
COPY scripts ./scripts
COPY src ./src
COPY models ./models
COPY rag_index ./rag_index

CMD ["python", "-m", "scripts.launch_dashboard", "--container"]
