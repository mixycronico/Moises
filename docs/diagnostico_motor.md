# Guía de Diagnóstico y Optimización del Motor Genesis

Este documento proporciona una guía para diagnosticar y solucionar problemas de rendimiento, bloqueos y comportamientos inesperados en el motor de eventos Genesis utilizando las pruebas de estrés implementadas.

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Pruebas de Estrés Disponibles](#pruebas-de-estrés-disponibles)
3. [Ejecutando las Pruebas](#ejecutando-las-pruebas)
4. [Interpretación de Resultados](#interpretación-de-resultados)
5. [Cuellos de Botella Comunes](#cuellos-de-botella-comunes)
6. [Estrategias de Optimización](#estrategias-de-optimización)
7. [Mejores Prácticas](#mejores-prácticas)

## Introducción

El motor de eventos Genesis es el núcleo del sistema de trading, responsable de la comunicación entre componentes y la ejecución de lógica de negocio. Su rendimiento y confiabilidad son críticos para el correcto funcionamiento del sistema.

Las pruebas de estrés permiten identificar:
- Cuellos de botella en el procesamiento de eventos
- Limitaciones en la expansión dinámica
- Problemas con el manejo de prioridades
- Dificultades con componentes lentos
- Comportamientos inesperados bajo alta carga

## Pruebas de Estrés Disponibles

El sistema incluye las siguientes pruebas de estrés específicas:

### 1. Aumento Gradual de Carga (`test_gradual_load_increase`)

Esta prueba va incrementando progresivamente el número de eventos por segundo hasta encontrar el punto donde el sistema comienza a degradarse. Permite identificar:
- Límite máximo de eventos por segundo
- Punto de inflexión donde la tasa de éxito cae
- Comportamiento del sistema cerca de sus límites

### 2. Aislamiento de Componentes Lentos (`test_slow_component_isolation`)

Verifica si el sistema puede aislar adecuadamente componentes lentos sin que afecten al rendimiento global. Permite identificar:
- Propagación de la latencia entre componentes
- Efectividad del aislamiento
- Impacto de componentes problemáticos en el resto del sistema

### 3. Alta Concurrencia (`test_high_concurrency`)

Simula un sistema con gran número de componentes emitiendo y recibiendo eventos simultáneamente. Permite identificar:
- Capacidad para manejar muchos componentes simultáneamente
- Distribución de carga entre componentes
- Comportamiento bajo patrones de tráfico realistas

### 4. Prioridades Bajo Presión (`test_priority_under_pressure`)

Verifica si el motor respeta las prioridades de eventos cuando está sometido a alta presión. Permite identificar:
- Respeto de prioridades bajo carga
- Diferencias de latencia entre niveles de prioridad
- Comportamiento del sistema cuando está saturado

### 5. Expansión Dinámica Acelerada (`test_dynamic_expansion_stress`)

Provoca cambios rápidos en la carga para forzar la expansión y contracción dinámica del motor. Permite identificar:
- Velocidad de adaptación del motor
- Eficiencia del escalado dinámico
- Correlación entre bloques activos y rendimiento

## Ejecutando las Pruebas

Las pruebas pueden ejecutarse individualmente o como un conjunto utilizando el script `scripts/run_stress_tests.py`:

```bash
# Ejecutar todas las pruebas
python scripts/run_stress_tests.py --all

# Ejecutar una prueba específica
python scripts/run_stress_tests.py --gradual

# Ejecutar varias pruebas específicas
python scripts/run_stress_tests.py --isolation --priority

# Especificar un archivo para el informe
python scripts/run_stress_tests.py --all --report-file=mi_informe.json
```

Los resultados detallados se guardan en formato JSON y también se muestra un resumen en la consola.

## Interpretación de Resultados

### Aumento Gradual de Carga

Principales métricas a analizar:
- **Eventos por segundo (EPS)**: El número máximo de eventos procesados por segundo antes de degradación
- **Tasa de éxito**: Porcentaje de eventos procesados exitosamente
- **Eficiencia de escalado**: Cómo escala el rendimiento con el tamaño del lote (idealmente lineal)

Diagnóstico:
- Si la tasa de éxito cae gradualmente, indica saturación gradual
- Si la tasa de éxito cae bruscamente, indica un punto de quiebre específico
- Si la eficiencia de escalado es baja (<0.5), indica problemas de escalabilidad

### Aislamiento de Componentes Lentos

Principales métricas a analizar:
- **Factor de impacto**: Cuánto afecta un componente lento al tiempo total
- **Diferencia de tiempo entre categorías**: Comparación entre componentes rápidos, medios y lentos

Diagnóstico:
- Factor de impacto >2.0: Aislamiento deficiente
- Factor de impacto <1.5: Buen aislamiento
- Si los componentes rápidos ven su rendimiento significativamente reducido, indica problemas de aislamiento

### Alta Concurrencia

Principales métricas a analizar:
- **Distribución de carga**: Desviación estándar y coeficiente de variación
- **Rendimiento total**: Eventos por segundo con 100 componentes

Diagnóstico:
- Alto coeficiente de variación (>0.5): Distribución de carga desigual
- Bajo rendimiento con muchos componentes: Posible problema de escalabilidad
- Alto número de timeouts: Posible sobrecarga del motor

### Prioridades Bajo Presión

Principales métricas a analizar:
- **Ratios de tiempo entre prioridades**: Cuánto más rápido se procesan los eventos de alta prioridad
- **Tasa de éxito por prioridad**: Si los eventos de alta prioridad tienen mayor tasa de éxito

Diagnóstico:
- Ratio bajo (<1.2) entre baja y alta prioridad: El sistema no respeta adecuadamente las prioridades
- Ratio alto (>2.0): Buen manejo de prioridades
- Tasa de éxito similar entre prioridades: Posible problema en el manejo de prioridades

### Expansión Dinámica Acelerada

Principales métricas a analizar:
- **Ratio de expansión máxima**: Cuánto se expande el motor bajo carga
- **Correlación bloques/rendimiento**: Si el aumento de bloques mejora el rendimiento
- **Velocidad de adaptación**: Cómo responde el motor a cambios bruscos de carga

Diagnóstico:
- Correlación <0.3: Expansión inefectiva, los bloques adicionales no mejoran el rendimiento
- Correlación >0.7: Expansión efectiva
- Adaptación lenta a cambios bruscos: Ajustar parámetros de escalado

## Cuellos de Botella Comunes

### 1. Saturación de Cola de Eventos

**Síntomas:**
- Caída abrupta de rendimiento en `test_gradual_load_increase`
- Alta tasa de timeouts en todas las pruebas

**Soluciones:**
- Aumentar tamaño de colas de eventos
- Implementar throttling adaptativo
- Mejorar mecanismos de backpressure

### 2. Overhead de Comunicación

**Síntomas:**
- Rendimiento degradado en `test_high_concurrency`
- Alto tiempo de respuesta incluso con pocos eventos

**Soluciones:**
- Reducir frecuencia de comunicación entre componentes
- Agrupar mensajes pequeños (batching)
- Optimizar serialización/deserialización

### 3. Bloqueo por Componentes Lentos

**Síntomas:**
- Alto factor de impacto en `test_slow_component_isolation`
- Componentes rápidos ralentizados por lentos

**Soluciones:**
- Mejorar el aislamiento entre componentes
- Implementar colas separadas por componente
- Agregar detección y mitigación de componentes problemáticos

### 4. Ineficiencia en Escalado Dinámico

**Síntomas:**
- Baja correlación bloques/rendimiento en `test_dynamic_expansion_stress`
- Expansión excesiva sin mejora de rendimiento

**Soluciones:**
- Ajustar umbrales de escalado
- Optimizar decisiones de asignación de eventos
- Implementar métricas más precisas para decidir escalado

### 5. Desorden en Prioridades

**Síntomas:**
- Ratio bajo entre prioridades en `test_priority_under_pressure`
- Eventos de baja prioridad procesados antes que los de alta

**Soluciones:**
- Revisar la implementación de colas de prioridad
- Agregar expiración para eventos de baja prioridad
- Implementar interrupción para eventos de alta prioridad

## Estrategias de Optimización

### Optimizaciones a Nivel de Motor

1. **Reducir Contención de Recursos**
   - Implementar colas múltiples para reducir contención
   - Utilizar estructuras de datos lock-free donde sea posible
   - Minimizar secciones críticas en el código

2. **Mejorar Throughput con Batching**
   - Procesar eventos en lotes cuando sea posible
   - Implementar coalescing para eventos similares
   - Optimizar la serialización/deserialización

3. **Afinar Parámetros de Escalado**
   - Ajustar umbral de escalado basado en pruebas
   - Reducir tiempo de enfriamiento si la carga fluctúa rápidamente
   - Implementar predicción de carga para escalado proactivo

### Optimizaciones a Nivel de Componente

1. **Reducir Latencia de Procesamiento**
   - Optimizar algoritmos en puntos críticos
   - Implementar caché para resultados frecuentes
   - Minimizar operaciones bloqueantes

2. **Mejorar Aislamiento**
   - Diseñar componentes con timeouts internos
   - Implementar circuit breakers para componentes inestables
   - Usar health checks proactivos

## Mejores Prácticas

### Desarrollo

1. **Componentes Resistentes a Fallos**
   - Diseñar componentes con capacidad de reinicio limpio
   - Implementar state recovery para recuperación tras fallos
   - Usar pattern retrying para operaciones transitorias

2. **Diseño para Escalabilidad**
   - Minimizar estado compartido entre componentes
   - Diseñar componentes stateless donde sea posible
   - Utilizar comunicación asíncrona de manera consistente

### Operación

1. **Monitoreo Continuo**
   - Capturar métricas de latencia, colas y éxito/fallo
   - Implementar alertas proactivas para degradación
   - Ejecutar pruebas de estrés periódicamente

2. **Troubleshooting**
   - Comenzar con pruebas aisladas y aumentar complejidad
   - Verificar componentes individualmente antes de end-to-end
   - Utilizar logs detallados para rastrear problemas

### Verificación

1. **Probar con Carga Realista**
   - Utilizar patrones de tráfico similares a producción
   - Incluir picos de carga y eventos irregulares
   - Simular condiciones adversas como latencia de red

2. **Validación Estadística**
   - Ejecutar múltiples pruebas para evitar resultados anómalos
   - Utilizar percentiles (p99, p999) en lugar de promedios
   - Analizar tendencias a lo largo del tiempo