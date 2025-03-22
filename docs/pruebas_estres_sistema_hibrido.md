# Pruebas de Estrés para el Sistema Híbrido API+WebSocket

## Introducción

Este documento describe las pruebas de estrés diseñadas para evaluar el rendimiento, la escalabilidad y la resiliencia del sistema híbrido API+WebSocket implementado en Genesis. Las pruebas están diseñadas para simular condiciones extremas y detectar posibles cuellos de botella, fugas de memoria y puntos de fallo antes de que ocurran en un entorno de producción.

## Objetivos de las Pruebas

Las pruebas de estrés tienen los siguientes objetivos específicos:

1. **Determinar límites de carga**: Identificar el volumen máximo de eventos y solicitudes que el sistema puede manejar manteniendo un rendimiento aceptable.

2. **Verificar resiliencia**: Comprobar que el sistema puede recuperarse de fallos en componentes individuales sin afectar al rendimiento global.

3. **Evaluar comportamiento bajo latencia**: Medir el impacto de condiciones de red adversas en el tiempo de respuesta y la fiabilidad del sistema.

4. **Detectar degradación progresiva**: Identificar si hay degradación de rendimiento o fugas de recursos durante operaciones prolongadas.

5. **Validar aislamiento de fallos**: Confirmar que los fallos en componentes específicos no causan fallos en cascada.

## Tipos de Pruebas Implementadas

### 1. Incremento del Volumen de Eventos

**Descripción**: Esta prueba aumenta gradualmente el número de eventos emitidos por segundo, desde algunas docenas hasta miles, hasta alcanzar los límites de rendimiento del sistema.

**Parámetros clave**:
- Tasa máxima de eventos por segundo
- Duración de la prueba
- Tiempo de escalado (ramp-up)

**Métricas medidas**:
- Eventos por segundo procesados
- Latencia de procesamiento
- Tasa de éxito
- Uso de memoria

**Criterios de éxito**:
- El sistema debe mantener una tasa de éxito >99% hasta 500 eventos/segundo
- La latencia media debe permanecer por debajo de 100ms

### 2. Variedad de Tipos de Eventos

**Descripción**: Esta prueba introduce varios tipos de eventos con diferentes características, incluyendo eventos con grandes volúmenes de datos, eventos de alta prioridad y eventos con dependencias entre sí.

**Parámetros clave**:
- Proporción de eventos grandes vs. pequeños
- Prioridades de eventos
- Patrones de dependencia

**Métricas medidas**:
- Latencia por tipo de evento
- Orden de procesamiento de eventos prioritarios
- Uso de memoria durante procesamiento de eventos grandes

**Criterios de éxito**:
- Los eventos prioritarios deben procesarse primero
- El sistema debe manejar eventos de hasta 1MB de tamaño sin degradación significativa

### 3. Simulación de Fallos en Tiempo Real

**Descripción**: Esta prueba simula fallos en componentes específicos mientras el sistema está en funcionamiento, para evaluar la capacidad de aislar fallos y mantener el servicio.

**Parámetros clave**:
- Componentes que fallarán
- Probabilidad y frecuencia de fallos
- Intervalo de recuperación

**Métricas medidas**:
- Tiempo hasta la recuperación
- Impacto en el rendimiento global durante fallos
- Fallos en cascada (si ocurren)

**Criterios de éxito**:
- Ningún fallo debe propagarse más allá del componente afectado
- El rendimiento global no debe degradarse más del 20% durante fallos

### 4. Condiciones de Latencia de Red Simulada

**Descripción**: Esta prueba introduce retrasos y fallos de red artificiales para evaluar cómo se comporta el sistema en condiciones de red adversas.

**Parámetros clave**:
- Perfiles de latencia (normal, lenta, muy lenta)
- Tasas de fallos de red
- Cambios dinámicos en condiciones de red

**Métricas medidas**:
- Adaptabilidad del sistema a cambios de condiciones
- Impacto en la experiencia del usuario
- Eficacia de los mecanismos de timeout

**Criterios de éxito**:
- El sistema debe adaptarse automáticamente a las condiciones de red
- Los timeouts deben prevenir bloqueos indefinidos

### 5. Ejecución Extendida

**Descripción**: Esta prueba ejecuta el sistema durante un período prolongado bajo carga moderada para detectar fugas de memoria, degradación progresiva u otros problemas que solo aparecen con el tiempo.

**Parámetros clave**:
- Duración (horas/días)
- Nivel de carga constante
- Intervalo de reporte de métricas

**Métricas medidas**:
- Uso de memoria a lo largo del tiempo
- Rendimiento en diferentes momentos
- Estabilidad a largo plazo

**Criterios de éxito**:
- El uso de memoria debe estabilizarse y no crecer indefinidamente
- El rendimiento debe mantenerse consistente durante toda la prueba

## Implementación Técnica

Las pruebas están implementadas en Python utilizando `asyncio` para manejar la concurrencia. El código está estructurado en varias clases clave:

1. **`PerformanceMetrics`**: Clase para recopilar y calcular métricas de rendimiento.

2. **`StressTestComponent`**: Componente simulado que puede experimentar fallos, latencia y crasheos controlados.

3. **`StressTestCoordinator`**: Coordinador que gestiona múltiples componentes y simula condiciones de red.

4. **Funciones de prueba**: Implementaciones específicas de cada tipo de prueba de estrés.

## Arquitectura de las Pruebas

El sistema de pruebas reproduce fielmente la arquitectura híbrida API+WebSocket:

```
┌──────────────────────────────────────────────────────┐
│                StressTestCoordinator                  │
├──────────────────┬───────────────────────────────────┤
│   API Requests   │        WebSocket Events           │
│  (with timeout)  │       (pub/sub model)             │
└─────────┬────────┴────────────────┬──────────────────┘
          │                         │
┌─────────▼─────────┐    ┌──────────▼──────────┐
│  Direct Requests  │    │   Event Delivery    │
│   to Components   │    │   to Subscribers    │
└─────────┬─────────┘    └──────────┬──────────┘
          │                         │
┌─────────▼─────────────────────────▼──────────┐
│             StressTestComponents             │
│                                              │
│  ┌─────────┐  ┌─────────┐      ┌─────────┐  │
│  │ Comp_0  │  │ Comp_1  │ ...  │ Comp_n  │  │
│  └─────────┘  └─────────┘      └─────────┘  │
└──────────────────────────────────────────────┘
```

## Planificación de Pruebas

Para una evaluación completa, se recomienda ejecutar las pruebas en el siguiente orden:

1. **Pruebas básicas**: Verificar funcionalidad inicial con cargas ligeras.
2. **Pruebas de volumen**: Determinar límites básicos de escalabilidad.
3. **Pruebas de tipos de eventos**: Evaluar respuesta a diferentes tipos de carga.
4. **Pruebas de fallos**: Verificar aislamiento y recuperación.
5. **Pruebas de red**: Evaluar comportamiento bajo condiciones adversas.
6. **Pruebas extendidas**: Verificar estabilidad a largo plazo.

## Análisis de Resultados

Los resultados de las pruebas se analizan para:

1. **Identificar cuellos de botella**: Componentes que limitan el rendimiento global.
2. **Detectar puntos de fallo**: Áreas donde el sistema es más vulnerable.
3. **Establecer umbrales seguros**: Determinar límites operativos recomendados.
4. **Recomendar optimizaciones**: Sugerir mejoras basadas en datos concretos.

## Recomendaciones de Uso

Para obtener resultados significativos:

1. Ejecutar pruebas en un entorno aislado que replique la producción lo más fielmente posible.
2. Comenzar con cargas ligeras y aumentar gradualmente la intensidad.
3. Mantener registros detallados de cada ejecución para análisis comparativo.
4. Realizar pruebas después de cada cambio significativo en la arquitectura.

## Conclusión

Las pruebas de estrés proporcionan una comprensión valiosa del comportamiento del sistema híbrido bajo condiciones extremas. Estas pruebas nos permiten identificar y abordar problemas potenciales antes de que afecten a los usuarios en un entorno de producción, garantizando así la fiabilidad y escalabilidad del sistema Genesis.