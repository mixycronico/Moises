# Sistema de Resiliencia Genesis

## Resumen Ejecutivo

El sistema de resiliencia Genesis es una capa de protección integral diseñada para mejorar la robustez, disponibilidad y recuperabilidad del sistema de trading, especialmente en condiciones adversas o no predecibles. Este documento describe las tres características principales que componen este sistema.

## Características Principales

### 1. Sistema de Reintentos Adaptativos

#### Descripción
Mecanismo que permite reintentar operaciones fallidas de forma inteligente, ajustando dinámicamente los intervalos entre intentos.

#### Implementación
```python
async def with_retry(func, max_retries=3, base_delay=0.1):
    attempt = 0
    while True:
        try:
            return await func()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            
            delay = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.1)
            total_delay = delay + jitter
            
            logger.info(f"Reintento {attempt}/{max_retries} tras error: {e}. Esperando {total_delay:.2f}s")
            await asyncio.sleep(total_delay)
```

#### Características
- **Backoff Exponencial**: Incremento progresivo del tiempo entre reintentos (fórmula: base_delay * 2^intento)
- **Jitter Aleatorio**: Variación aleatoria añadida para evitar sincronización de reintentos
- **Límite Configurable**: Número máximo de reintentos personalizable
- **Registro Detallado**: Información completa sobre intentos y errores

#### Ventajas
- Reduce la presión inmediata sobre componentes fallidos
- Permite recuperación automática de fallos transitorios
- Previene efectos de manada en reintentos simultáneos

### 2. Patrón Circuit Breaker

#### Descripción
Sistema inspirado en los fusibles eléctricos que aísla componentes fallidos para prevenir fallos en cascada.

#### Implementación
```python
class CircuitBreaker:
    def __init__(self, name, failure_threshold=3, recovery_timeout=5.0):
        self.name = name
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_timeout
        self.last_failure_time = 0
    
    async def execute(self, func):
        # Lógica del circuit breaker
        # ...
```

#### Estados
- **CLOSED**: Estado normal, las operaciones se ejecutan directamente
- **OPEN**: Fallo detectado, las operaciones se rechazan inmediatamente
- **HALF-OPEN**: Estado probatorio que permite un número limitado de operaciones para verificar recuperación

#### Transiciones
- **CLOSED → OPEN**: Cuando se detectan N fallos consecutivos
- **OPEN → HALF-OPEN**: Después de un tiempo de espera configurable
- **HALF-OPEN → CLOSED**: Cuando se completan M operaciones exitosas consecutivas
- **HALF-OPEN → OPEN**: Si ocurre un solo fallo durante el período probatorio

#### Ventajas
- Aísla rápidamente componentes problemáticos
- Previene la propagación de fallos a través del sistema
- Permite recuperación gradual y controlada
- Protege recursos del sistema durante períodos de inestabilidad

### 3. Sistema de Checkpointing y Recuperación

#### Descripción
Mecanismo que preserva el estado crítico del sistema y permite restaurarlo rápidamente tras fallos.

#### Implementación
```python
class CheckpointSystem:
    def __init__(self, component_id, checkpoint_dir, auto_checkpoint=True):
        self.component_id = component_id
        self.checkpoint_dir = checkpoint_dir
        self.state = {}
        self.checkpoints = {}
        self.auto_checkpoint = auto_checkpoint
    
    async def create_checkpoint(self):
        # Implementación del checkpoint
        # ...
    
    async def restore_latest(self):
        # Restauración desde checkpoint
        # ...
```

#### Características
- **Checkpointing Automático**: Creación periódica de puntos de recuperación
- **Checkpointing Manual**: Posibilidad de crear checkpoints en momentos críticos
- **Recuperación Selectiva**: Capacidad de restaurar componentes específicos
- **Modo Seguro**: Estado operativo reducido para componentes críticos
- **Metadata de Checkpoints**: Información completa sobre cada punto de recuperación

#### Ventajas
- Minimiza pérdida de datos tras fallos
- Reduce tiempo de recuperación
- Proporciona puntos conocidos y estables de restauración
- Permite operación parcial en condiciones degradadas

## Integración de Características

El verdadero poder del sistema de resiliencia Genesis radica en la integración de estas tres características, que trabajan juntas para proporcionar una defensa en profundidad:

1. El **Sistema de Reintentos** maneja fallos temporales o transitorios
2. Si los fallos persisten, el **Circuit Breaker** aísla el componente problemático
3. Para fallos graves, el **Sistema de Checkpointing** permite recuperar el estado rápidamente

## Modo Degradado (Safe Mode)

Cuando múltiples componentes fallan, el sistema puede entrar en modo degradado:

- **Modo SAFE**: Se priorizan componentes esenciales y operaciones de sólo lectura
- **Modo EMERGENCY**: Solo operan componentes marcados como críticos

## Métricas de Resiliencia

El sistema proporciona métricas en tiempo real sobre su estado de resiliencia:

- Tasa de reintentos y éxito/fallo
- Estado de todos los circuit breakers
- Disponibilidad y edad de checkpoints
- Modo actual del sistema (NORMAL/SAFE/EMERGENCY)

## Conclusión

El sistema de resiliencia Genesis proporciona una capa robusta de protección que permite:

- Mantener operación parcial durante fallos
- Aislar y contener problemas antes de que afecten a todo el sistema
- Recuperarse rápidamente tras interrupciones
- Degradar servicios de forma controlada cuando sea necesario

Con estas características, la fiabilidad general del sistema se incrementará significativamente, pasando del actual 71% a una meta objetivo superior al 90%.