# The geo risk API on its own, for deploying somewhere other than a laptop.
# Build from the talk directory: `docker build -f geo_service/Dockerfile -t geo-risk-api .`
FROM python:3.14-slim

WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.115.0" "uvicorn>=0.34.0" "shapely>=2.0.0" "logfire[fastapi]>=5.4.0"

COPY geo_service/__init__.py geo_service/app.py geo_service/geo.py geo_service/
COPY geo_service/data geo_service/data

EXPOSE 8001
CMD ["uvicorn", "geo_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
