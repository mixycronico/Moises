# Reporte de Errores del Sistema Genesis

## 1. Errores de Importación

### 1.1 Clase `EnginePriorityBlocks` vs `PriorityBlockEngine`

Se ha detectado una inconsistencia en la nomenclatura de la clase de motor de eventos con bloques prioritarios:

| Archivo | Problema |
|---------|----------|
| `scripts/run_stress_tests.py` | Referencia a `EnginePriorityBlocks` en lugar de `PriorityBlockEngine` |
| `tests/unit/core/test_core_stress_tests.py` | Importación incorrecta de `EnginePriorityBlocks` |
| `tests/unit/core/conftest.py` | Comentario referenciando a `EnginePriorityBlocks` |

### 1.2 Errores de Módulos No Encontrados

Múltiples archivos presentan errores de importación para módulos estándar (`pytest`, `numpy`, `pandas`) y módulos internos del sistema:

| Módulo | Archivos Afectados |
|--------|-------------------|
| `pytest` | Mayoría de archivos de prueba |
| `numpy` | Archivos de prueba de análisis de datos |
| `pandas` | Archivos de prueba de análisis de datos |
| `genesis.data.*` | `tests/unit/data/test_data_manager_advanced.py` |
| `genesis.execution.*` | `tests/unit/execution/test_execution_advanced.py` |
| `genesis.risk.*` | `tests/unit/risk/test_risk_manager_advanced.py` |
| `genesis.exchange.*` | `tests/integration/test_system_integration.py` |

## 2. Errores de Tipo

### 2.1 Problemas con Valores `None`

| Archivo | Problema |
|---------|----------|
| `tests/unit/core/test_core_extreme_scenarios_optimized.py` | Línea 44: `Expression of type "None" cannot be assigned to parameter of type "List[str]"` |
| `tests/unit/core/test_core_race_conditions.py` | Línea 45: `Expression of type "None" cannot be assigned to parameter of type "List[str]"` |
| `tests/unit/core/test_core_peak_load_recovery.py` | Línea 296: `Expression of type "None" cannot be assigned to parameter of type "List[str]"` |

### 2.2 Incompatibilidad de Tipos en Métodos Sobrescritos

| Archivo | Problema |
|---------|----------|
| `tests/unit/core/test_core_intermediate.py` | Múltiples métodos `start` y `stop` devuelven `Literal[True]` pero se esperaba `None` |
| `tests/unit/core/test_core_intermediate_optimized.py` | Mismos problemas con los métodos `start` y `stop` |

### 2.3 Errores en Tipos de Argumentos

| Archivo | Problema |
|---------|----------|
| `tests/unit/core/test_engine_start_stop.py` | `Argument of type "SimpleComponent" cannot be assigned to parameter "component" of type "Component"` |
| `tests/unit/core/test_engine_ultra_minimal.py` | `Argument of type "MinimalComponent" cannot be assigned to parameter "component" of type "Component"` |
| `tests/unit/core/test_core_stress_tests.py` | Línea 820: `Argument of type "float" cannot be assigned to parameter "value" of type "int"` |

## 3. Errores de Acceso a Miembros

### 3.1 Objetos `None` Usados Incorrectamente

| Archivo | Problema |
|---------|----------|
| `genesis/core/event_bus.py` | Línea 186: `"event_type" is possibly unbound` |
| `tests/unit/core/test_core_advanced.py` | Línea 65: `"emit" is not a known member of "None"` |
| `tests/unit/core/test_core_event_bus_simplified.py` | Línea 44: `Object of type "None" is not subscriptable` |
| `tests/unit/core/test_core_intermediate.py` | Líneas 133, 203: `"emit" is not a known member of "None"` |
| `tests/unit/core/test_core_intermediate_optimized.py` | Líneas 138, 209: `"emit" is not a known member of "None"` |
| `tests/unit/risk/test_risk_manager_advanced.py` | Múltiples líneas: `"emit" is not a known member of "None"` y `Object of type "None" is not subscriptable` |

### 3.2 Miembros No Reconocidos en Clases

| Archivo | Problema |
|---------|----------|
| `tests/unit/core/test_core_priority_alternative.py` | Líneas 82-84: `Cannot access member "call_count" for type "function"` |
| `tests/unit/core/test_core_extreme_optimized.py` | `Cannot access member "enable_advanced_recovery" for type "ConfigurableTimeoutEngine"` |
| `tests/unit/risk/test_risk_manager_advanced.py` | `Cannot access member "get_risk_percentage" for type "PositionSizer"` |
| `tests/unit/core/test_core_peak_load_recovery.py` | `Cannot access member "_engine" for type "BurstMonitorComponent*"` |

## 4. Errores de Parámetros

### 4.1 Parámetros No Reconocidos

| Archivo | Problema |
|---------|----------|
| `tests/unit/core/conftest.py` | Líneas 49-51: `No parameter named "default_timeout"`, `"handler_timeout"`, `"recovery_timeout"` |
| `tests/unit/core/conftest.py` | Líneas 79-82: `No parameter named "min_blocks"`, `"max_blocks"`, `"scaling_threshold"`, `"cooldown_period"` |
| `tests/unit/core/test_core_peak_load_recovery.py` | Líneas 553, 913-916: Mismos problemas con parámetros |
| `scripts/generate_load_recovery_report.py` | Líneas 96-99, 119-122: Mismos problemas con parámetros |

### 4.2 Errores en Número de Argumentos

| Archivo | Problema |
|---------|----------|
| `scripts/generate_load_recovery_report.py` | Líneas 107, 130: `Expected 0 positional arguments` |

## 5. Resumen de Acciones Correctivas

1. **Corregidos:**
   - Renombrar `EnginePriorityBlocks` a `PriorityBlockEngine` en archivos clave

2. **Pendientes:**
   - Corregir errores de tipo en componentes y clases
   - Asegurar que los métodos sobrescritos coincidan con la firma de la clase base
   - Corregir accesos a miembros en objetos que pueden ser None
   - Agregar verificaciones de nulidad en funciones clave
   - Corregir parámetros faltantes o incorrectos
   - Revisar las importaciones de módulos externos

## 6. Priorización de Correcciones

1. **Alta Prioridad:**
   - Errores de importación que bloquean la ejecución de pruebas críticas
   - Errores donde se utilizan objetos None de manera incorrecta
   - Errores en el motor principal (core engine)

2. **Media Prioridad:**
   - Inconsistencias en tipos de parámetros
   - Errores en tests avanzados/opcionales
   - Errores en clases heredadas

3. **Baja Prioridad:**
   - Comentarios desactualizados
   - Advertencias sobre importaciones no utilizadas
   - Parámetros opcionales no reconocidos