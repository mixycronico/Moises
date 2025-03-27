FROM python:3.9

WORKDIR /app

# Copiar archivos de dependencias
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir psycopg2-binary websocket-client tensorflow flask tenacity numpy pandas matplotlib ccxt flask-cors gunicorn

# Puerto para la aplicación Flask
EXPOSE 5000

# Comando para iniciar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--reuse-port", "--reload", "main:app"]