FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# bake data + models into the image so containers start instantly (use `make data-tlc train` for real data)
RUN python scripts/prepare_data.py --source synthetic && python scripts/train_models.py
EXPOSE 8000 8765
CMD ["uvicorn", "surgeshield.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
