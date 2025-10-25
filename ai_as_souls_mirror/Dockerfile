FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8710
CMD ["uvicorn","mirror_experience:app","--host","0.0.0.0","--port","8710"]
