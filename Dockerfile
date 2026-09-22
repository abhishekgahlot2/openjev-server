FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY openjev_server ./openjev_server
COPY profiles ./profiles
RUN pip install --no-cache-dir .
EXPOSE 3000
ENV OPENJEV_HOST=0.0.0.0 OPENJEV_PORT=3000 OPENJEV_BACKEND=vllm OPENJEV_VLLM_URL=http://vllm:8000/v1 OPENJEV_MODEL=openjev/openjev-FP8 OPENJEV_PROFILE=openjev
CMD ["openjev", "serve"]
