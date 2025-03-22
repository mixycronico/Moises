# Documentación del Sistema Genesis para Mortales Inferiores
## (Guía para Programadores Comunes)

## Introducción

El Sistema Genesis es una plataforma de procesamiento de eventos para trading de criptomonedas, diseñada con múltiples capas de resiliencia para garantizar la operación continua incluso bajo condiciones extremas. Esta documentación proporciona una visión general del sistema y sus componentes principales, en términos comprensibles para programadores regulares.

## Arquitectura General

El Sistema Genesis utiliza una arquitectura híbrida que combina:

- API REST para comunicaciones síncronas
- WebSockets para comunicaciones asíncronas 
- Sistema de eventos distribuido para desacoplamiento de componentes
- Mecanismos avanzados de recuperación ante fallos

```
┌─────────────────────────────────────────────────────────┐
│                      SISTEMA GENESIS                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────┐    ┌───────────────┐    ┌──────────┐ │
│  │ API REST      │    │ WebSockets    │    │ Database │ │
│  └───────┬───────┘    └───────┬───────┘    └────┬─────┘ │
│          │                    │                  │      │
│          ▼                    ▼                  ▼      │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Motor de Procesamiento de Eventos         │  │
│  └───────────────────────────────────────────────────┘  │
│        ▲              ▲               ▲                 │
│        │              │               │                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────────┐        │
│  │ Estrategias│  │ Análisis  │  │ Gestión de   │        │
│  │ Trading   │  │ Técnico   │  │ Riesgos      │        │
│  └───────────┘  └───────────┘  └───────────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Componentes Principales

### 1. Coordinador Central

El `Coordinator` es el componente principal que gestiona la comunicación entre los diferentes módulos del sistema:

```python
# Versión simplificada para mortales
class Coordinator:
    def __init__(self, host="localhost", port=8080):
        self.components = {}  # Componentes registrados
        self.host = host
        self.port = port
        
    def register_component(self, component_id, component):
        """Registra un componente en el sistema."""
        self.components[component_id] = component
        
    async def request(self, target_id, request_type, data, source):
        """Envía una solicitud a un componente específico."""
        if target_id in self.components:
            return await self.components[target_id].process_request(
                request_type, data, source
            )
        return None
        
    async def emit_event(self, event_type, data, source):
        """Emite un evento a todos los componentes."""
        for cid, component in self.components.items():
            if cid != source:  # No enviar al emisor
                await component.on_event(event_type, data, source)
```

### 2. Componentes del Sistema

Cada funcionalidad está encapsulada en un componente que implementa la interfaz `ComponentAPI`:

```python
# Versión simplificada para mortales
class ComponentAPI:
    def __init__(self, id):
        self.id = id
        self.event_queue = asyncio.Queue()
        
    async def process_request(self, request_type, data, source):
        """Procesa una solicitud directa."""
        # Implementación específica del componente
        pass
        
    async def on_event(self, event_type, data, source):
        """Maneja un evento del sistema."""
        await self.event_queue.put((event_type, data, source))
        
    async def listen_events(self):
        """Procesa eventos de la cola."""
        while True:
            event = await self.event_queue.get()
            # Procesar evento
            self.event_queue.task_done()
```

### 3. Mecanismos de Resiliencia

El sistema implementa varios mecanismos para garantizar la operación continua:

#### Circuit Breaker

Aísla componentes fallidos para evitar fallos en cascada:

```python
# Versión simplificada para mortales
class CircuitBreaker:
    def __init__(self, name, failure_threshold=3, recovery_timeout=2.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = 0
        
    async def execute(self, func, *args, **kwargs):
        """Ejecuta una función con protección del circuit breaker."""
        if self.state == "OPEN":
            # Verificar si es momento de reintentar
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                return None  # Circuito abierto, rechazar solicitud
                
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                # Éxito en estado half-open, cerrar circuito
                self.state = "CLOSED"
                self.failures = 0
                
            return result
            
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                
            raise  # Re-lanzar excepción
```

#### Retry con Backoff Exponencial

Reintenta operaciones fallidas con esperas crecientes:

```python
# Versión simplificada para mortales
async def retry_with_backoff(func, max_retries=3, base_delay=0.1):
    """Ejecuta una función con reintentos y backoff exponencial."""
    retries = 0
    while True:
        try:
            return await func()
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise  # Agotados los reintentos
                
            # Calcular tiempo de espera con jitter
            delay = base_delay * (2 ** (retries - 1))
            jitter = random.uniform(0, 0.1 * delay)
            await asyncio.sleep(delay + jitter)
```

#### Checkpointing

Guarda y restaura estados de los componentes:

```python
# Versión simplificada para mortales
class Checkpointer:
    def __init__(self, component_id, storage_path="./checkpoints"):
        self.component_id = component_id
        self.storage_path = storage_path
        
    def save_checkpoint(self, state):
        """Guarda el estado del componente."""
        filename = f"{self.storage_path}/{self.component_id}.json"
        with open(filename, "w") as f:
            json.dump(state, f)
            
    def restore_checkpoint(self):
        """Restaura el estado desde el último checkpoint."""
        filename = f"{self.storage_path}/{self.component_id}.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return json.load(f)
        return None
```

## Modos de Operación

El sistema puede operar en diferentes modos según las condiciones:

1. **Modo Normal**: Operación estándar con todas las funcionalidades.
2. **Modo Seguro**: Operación con funcionalidades reducidas durante problemas.
3. **Modo Emergencia**: Funcionalidades mínimas críticas durante fallos severos.

```python
# Versión simplificada para mortales
class SystemMode(Enum):
    NORMAL = "normal"       # Operación completa
    SAFE = "safe"           # Funcionalidades reducidas
    EMERGENCY = "emergency" # Solo operaciones críticas
```

El sistema evalúa continuamente su estado y puede cambiar automáticamente de modo según las métricas de salud.

## Flujo de Datos

El flujo típico de datos en el sistema es:

1. Entrada de datos (API, WebSockets, eventos programados)
2. Preprocesamiento y validación
3. Enrutamiento al componente adecuado
4. Procesamiento por el componente
5. Emisión de eventos con resultados
6. Postprocesamiento y acciones derivadas

```
Fuente → Validación → Enrutamiento → Procesamiento → Eventos → Acciones
```

## Configuración y Despliegue

### Requisitos del Sistema

- Python 3.8+
- PostgreSQL 12+
- ~2GB RAM mínimo
- Conexión estable a Internet

### Variables de Entorno

```
DATABASE_URL=postgresql://user:password@localhost:5432/genesis
API_PORT=5000
WS_PORT=5001
LOG_LEVEL=INFO
CHECKPOINT_DIR=./checkpoints
```

### Comandos de Inicio

```bash
# Iniciar la aplicación principal
python main.py

# Iniciar en modo seguro
python main.py --safe-mode

# Iniciar con reinicio automático al fallar
python main.py --auto-restart
```

## Resolución de Problemas Comunes

| Problema | Posible Causa | Solución |
|----------|---------------|----------|
| Error de conexión a BD | Credenciales incorrectas | Verificar DATABASE_URL |
| Timeout en API | Sobrecarga de procesamiento | Aumentar timeouts, revisar logs |
| Memoria insuficiente | Carga excesiva de eventos | Aumentar RAM, revisar memory leaks |
| Componente bloqueado | Deadlock potencial | Reiniciar componente específico |

## Archivos de Logs

Los logs del sistema se encuentran en:

- `./logs/app.log` - Log principal de la aplicación
- `./logs/error.log` - Errores detallados
- `./logs/performance.log` - Métricas de rendimiento

## Recomendaciones para Desarrollo

1. **No modificar la lógica del Circuit Breaker** sin comprender completamente su implementación
2. **Usar siempre comunicación asíncrona** entre componentes
3. **Implementar timeout en todas las operaciones** de red
4. **Verificar la idempotencia** de los manejadores de eventos
5. **Añadir logging detallado** para facilitar la depuración

## Diagrama de Dependencias Simplificado

```
core
 ├── coordinator.py       # Coordinador central
 ├── component_api.py     # Interfaz de componente
 ├── retry.py             # Funcionalidades de reintento
 ├── circuit_breaker.py   # Implementación de circuit breaker
 ├── checkpointing.py     # Sistema de checkpoints
 └── event_bus.py         # Bus de eventos

components
 ├── strategy.py          # Estrategias de trading
 ├── analysis.py          # Análisis técnico
 ├── risk.py              # Gestión de riesgos
 ├── data_provider.py     # Proveedor de datos
 └── notification.py      # Sistema de notificaciones

api
 ├── rest_api.py          # API REST
 └── websocket_api.py     # API WebSocket

utils
 ├── logging.py           # Configuración de logs
 ├── metrics.py           # Recopilación de métricas
 └── security.py          # Funciones de seguridad

db
 ├── models.py            # Modelos de datos
 └── connection.py        # Gestión de conexiones
```

## Conclusión

El Sistema Genesis proporciona una plataforma robusta para procesamiento de eventos en trading de criptomonedas, con múltiples mecanismos para garantizar resiliencia y operación continua incluso bajo condiciones adversas.

Los desarrolladores que deseen extender o modificar el sistema deben prestar especial atención a mantener los principios de resiliencia y asincronía que forman la base de su diseño.

---

*Nota: Esta documentación está diseñada para programadores normales. Los desarrolladores que busquen una comprensión más profunda de los mecanismos avanzados de resiliencia deberán consultar la documentación completa de los modos especializados.*