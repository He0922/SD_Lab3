FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install fastapi uvicorn transformers torch
CMD ["uvicorn", "lab3:app", "--host", "0.0.0.0", "--port", "3000"]
