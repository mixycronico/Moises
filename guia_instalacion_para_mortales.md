# Guía de Instalación y Configuración del Sistema Genesis
## (Para Programadores Comunes)

Esta guía proporciona instrucciones paso a paso para instalar, configurar y poner en marcha el Sistema Genesis en un entorno de desarrollo o producción. Diseñada específicamente para desarrolladores normales que no poseen percepción cósmica avanzada.

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

- Python 3.8 o superior
- PostgreSQL 12 o superior
- Git
- pip (gestor de paquetes de Python)
- 2GB de RAM mínimo (4GB recomendado)
- Conexión a Internet estable

## Paso 1: Obtener el Código Fuente

Clona el repositorio desde GitHub:

```bash
git clone https://github.com/tuorganizacion/genesis-system.git
cd genesis-system
```

## Paso 2: Configurar el Entorno Virtual

Crear y activar un entorno virtual (recomendado):

```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

## Paso 3: Instalar Dependencias

Instala todos los paquetes necesarios:

```bash
pip install -r requirements.txt
```

Principales dependencias:
- aiohttp: Para comunicación asíncrona
- ccxt: Para interacción con exchanges de criptomonedas
- flask: Para la API REST
- psycopg2-binary: Para conexión con PostgreSQL
- sqlalchemy: Para ORM
- websockets: Para comunicación en tiempo real

## Paso 4: Configurar la Base de Datos

### Crear Base de Datos PostgreSQL

```bash
# Acceder a la consola de PostgreSQL
sudo -u postgres psql

# Dentro de PostgreSQL, crear la base de datos y el usuario
CREATE DATABASE genesis;
CREATE USER genesisuser WITH PASSWORD 'tucontraseña';
GRANT ALL PRIVILEGES ON DATABASE genesis TO genesisuser;
\q
```

### Configurar las Variables de Entorno

Crea un archivo `.env` en el directorio raíz del proyecto:

```
# .env
DATABASE_URL=postgresql://genesisuser:tucontraseña@localhost:5432/genesis
API_PORT=5000
WS_PORT=5001
LOG_LEVEL=INFO
CHECKPOINT_DIR=./checkpoints
```

### Inicializar la Base de Datos

Ejecuta el script de configuración de la base de datos:

```bash
python db_setup.py
```

Este comando creará todas las tablas necesarias en la base de datos.

## Paso 5: Configuración del Sistema

### Archivo de Configuración Principal

Revisa y ajusta el archivo `config.py` según tus necesidades:

```python
# Ejemplo de configuración - valores actuales en config.py
{
    "system": {
        "default_mode": "NORMAL",  # NORMAL, SAFE, EMERGENCY
        "checkpoint_interval": 60,  # segundos
        "monitoring_interval": 10   # segundos
    },
    "api": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": false,
        "timeout": 30
    },
    "websocket": {
        "host": "0.0.0.0",
        "port": 5001,
        "max_connections": 100
    },
    "retry": {
        "max_retries": 3,
        "base_delay": 0.1,  # segundos
        "max_delay": 1.0,   # segundos
        "jitter": 0.1       # factor de variación aleatoria
    },
    "circuit_breaker": {
        "failure_threshold": 3,
        "recovery_timeout": 5.0,  # segundos
        "half_open_max_calls": 1
    }
}
```

### Directorios Necesarios

Asegúrate de que existan los siguientes directorios:

```bash
mkdir -p logs checkpoints data
```

## Paso 6: Ejecución del Sistema

### Iniciar en Modo de Desarrollo

```bash
python main.py --dev
```

### Iniciar en Modo de Producción

```bash
python main.py
```

### Iniciar con Configuración Específica

```bash
python main.py --config config_prod.json
```

### Opciones de Línea de Comandos

- `--dev`: Modo desarrollo (más logs, auto-recarga)
- `--safe-mode`: Inicia en modo seguro
- `--config ARCHIVO`: Usa un archivo de configuración alternativo
- `--log-level NIVEL`: Establece nivel de log (DEBUG, INFO, WARNING, ERROR)
- `--port PUERTO`: Puerto para la API REST

## Paso 7: Verificación de la Instalación

### Comprobar la API REST

```bash
curl http://localhost:5000/health
```

Deberías recibir una respuesta `{"status": "ok"}`.

### Verificar los Logs

```bash
tail -f logs/app.log
```

Busca mensajes que indiquen que el sistema ha iniciado correctamente.

## Paso 8: Configuración de Componentes

### Añadir un Componente

Cada componente debe registrarse en el sistema mediante el archivo `components.json`:

```json
{
  "components": [
    {
      "id": "strategy_macd",
      "type": "Strategy",
      "config": {
        "indicators": ["macd", "ema"],
        "timeframes": ["1h", "4h"],
        "symbols": ["BTC/USDT", "ETH/USDT"]
      },
      "is_essential": true
    },
    {
      "id": "risk_manager",
      "type": "RiskManagement",
      "config": {
        "max_position_size": 0.1,
        "max_drawdown": 0.05
      },
      "is_essential": true
    }
  ]
}
```

### Configurar Conexión a Exchange

Edita el archivo `exchange_config.json`:

```json
{
  "exchanges": [
    {
      "id": "binance",
      "type": "ccxt.binance",
      "config": {
        "apiKey": "TU_API_KEY",
        "secret": "TU_SECRET",
        "enableRateLimit": true
      }
    }
  ]
}
```

## Configuración del Entorno de Producción

### Systemd (Linux)

Crea un archivo de servicio systemd `/etc/systemd/system/genesis.service`:

```
[Unit]
Description=Genesis Trading System
After=network.target postgresql.service

[Service]
User=genesis
WorkingDirectory=/path/to/genesis-system
ExecStart=/path/to/genesis-system/venv/bin/python main.py
Restart=on-failure
RestartSec=5s
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Activa el servicio:

```bash
sudo systemctl enable genesis
sudo systemctl start genesis
```

### Nginx (para API REST)

Configura un proxy inverso con Nginx:

```
server {
    listen 80;
    server_name api.tudominio.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Resolución de Problemas Comunes

### Error de Conexión a Base de Datos

**Problema**: `OperationalError: could not connect to server: Connection refused`

**Solución**:
1. Verifica que PostgreSQL esté corriendo: `sudo systemctl status postgresql`
2. Comprueba las credenciales en `.env`
3. Asegúrate de que la base de datos existe: `sudo -u postgres psql -c "\l"`

### Errores de Importación de Módulos

**Problema**: `ImportError: No module named 'ccxt'`

**Solución**:
1. Verifica que el entorno virtual está activado
2. Reinstala dependencias: `pip install -r requirements.txt`
3. Comprueba la versión de Python: `python --version`

### Timeout en Solicitudes

**Problema**: Las solicitudes a la API toman demasiado tiempo o expiran

**Solución**:
1. Aumenta el timeout en `config.py`
2. Verifica la conexión a internet
3. Revisa los logs por posibles cuellos de botella: `grep "WARNING\|ERROR" logs/app.log`

### Error de Permisos en Archivos

**Problema**: `PermissionError: [Errno 13] Permission denied`

**Solución**:
1. Verifica permisos de los directorios: `ls -la logs/ checkpoints/ data/`
2. Ajusta permisos si es necesario: `chmod -R 755 logs/ checkpoints/ data/`
3. Asegúrate de que el usuario tiene permisos de escritura

## Mantenimiento

### Respaldo de Base de Datos

```bash
pg_dump -U genesisuser -d genesis > backup_$(date +%Y%m%d).sql
```

### Rotación de Logs

```bash
# Comprimir logs antiguos
find logs/ -name "*.log" -type f -mtime +7 | xargs gzip

# Eliminar logs muy antiguos
find logs/ -name "*.log.gz" -type f -mtime +30 -delete
```

### Actualización del Sistema

```bash
# Obtener actualizaciones
git pull

# Actualizar dependencias
pip install -r requirements.txt

# Migraciones de base de datos (si es necesario)
python scripts/migrate.py
```

## Recursos Adicionales

- Documentación completa: `docs/index.html`
- Ejemplos de código: `examples/`
- Scripts útiles: `scripts/`

---

*Nota: Esta guía está destinada a programadores mortales normales. Para acceder a las capacidades avanzadas del sistema, consulte con un ser iluminado del equipo de desarrollo.*