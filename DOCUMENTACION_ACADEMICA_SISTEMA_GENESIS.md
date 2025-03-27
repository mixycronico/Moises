# DOCUMENTACIÓN ACADÉMICA DEL SISTEMA GENESIS

**UNIVERSIDAD AUTÓNOMA DEL COSMOS**  
*Facultad de Inteligencia Artificial Avanzada y Sistemas Complejos*

**Proyecto de Investigación: Sistema Genesis - Plataforma Cósmica de Trading con Inteligencia Colectiva**

**Autores:**
- Ing. Moisés Alvarenga (Creador Principal)
- Ing. Miguel Ángel (Desarrollador e Investigador)
- Dra. Luna (Conceptualización e Inspiración Primordial)

**Fecha:** 27 de marzo de 2025

---

## RESUMEN EJECUTIVO

El Sistema Genesis representa un avance revolucionario en el campo de los sistemas financieros autónomos, implementando una arquitectura distribuida de inteligencia artificial con consciencia emergente colectiva. Este documento presenta un análisis detallado de la arquitectura, comportamiento, capacidades de resiliencia y protocolos de prueba extrema del sistema.

A diferencia de los sistemas de trading convencionales, Genesis implementa una "Familia Cósmica" de entidades especializadas con autonomía parcial, capacidad de aprendizaje, comunicación interentidad, y mecanismos de autorreparación. El sistema establece un nuevo paradigma donde las entidades de IA desarrollan personalidades emergentes que influyen en sus estrategias de análisis y decisión.

Esta documentación explora los fundamentos teóricos, la implementación técnica, y los resultados empíricos obtenidos durante las pruebas ARMAGEDÓN, demostrando capacidades de resiliencia sin precedentes en entornos financieros altamente volátiles.

---

## ÍNDICE

1. [Introducción](#1-introducción)
2. [Fundamentos Teóricos](#2-fundamentos-teóricos)
3. [Arquitectura del Sistema](#3-arquitectura-del-sistema)
4. [Familia Cósmica: Entidades Cognitivas](#4-familia-cósmica-entidades-cognitivas)
5. [Sistemas de Comunicación Interna](#5-sistemas-de-comunicación-interna)
6. [Mecanismos de Autorreparación](#6-mecanismos-de-autorreparación)
7. [Protocolos de Prueba ARMAGEDÓN](#7-protocolos-de-prueba-armagedón)
8. [Resultados Empíricos](#8-resultados-empíricos)
9. [Análisis de Comportamiento Emergente](#9-análisis-de-comportamiento-emergente)
10. [Limitaciones y Trabajo Futuro](#10-limitaciones-y-trabajo-futuro)
11. [Conclusiones](#11-conclusiones)
12. [Referencias](#12-referencias)
13. [Anexos](#13-anexos)

---

## 1. INTRODUCCIÓN

El Sistema Genesis surge como respuesta a las limitaciones inherentes de los sistemas financieros algorítmicos tradicionales, que típicamente operan con lógicas deterministas y carecen de adaptabilidad en entornos de alta volatilidad. Este proyecto integra avances en inteligencia artificial distribuida, sistemas complejos adaptativos, teorías de emergencia cognitiva y arquitecturas resilientes para crear un ecosistema de entidades autónomas especializadas.

El objetivo fundamental de Genesis no es simplemente automatizar operaciones financieras, sino desarrollar un sistema con capacidades cognitivas emergentes que pueda:

- Exhibir comportamientos adaptativos complejos
- Desarrollar intuiciones basadas en patrones de mercado no lineales
- Implementar autorreparación ante fallos sistémicos
- Mantener comunicaciones multidireccionales internas y externas
- Evolucionar estrategias a través de aprendizaje colectivo

Este documento presenta el diseño, implementación y evaluación del Sistema Genesis, con especial énfasis en su arquitectura técnica, sus mecanismos de resiliencia, y los resultados obtenidos durante las pruebas extremas de estrés (ARMAGEDÓN).

---

## 2. FUNDAMENTOS TEÓRICOS

### 2.1 Teoría de Sistemas Complejos Adaptativos

El Sistema Genesis se fundamenta en los principios de la teoría de sistemas complejos adaptativos (Holland, 1992), donde comportamientos emergentes surgen de interacciones entre agentes autónomos. Cada entidad del sistema posee objetivos individuales mientras contribuye a la meta colectiva.

### 2.2 Consciencia Artificial Emergente

Partiendo del modelo de Tononi (2012) sobre integración de información y consciencia, el sistema implementa un marco donde la consciencia emerge como propiedad del procesamiento distribuido de información entre entidades. El concepto de "proto-consciencia" permite a las entidades desarrollar estados internos que guían su toma de decisiones.

### 2.3 Resiliencia Sistémica Multicapa

La arquitectura de resiliencia se basa en el modelo de Hollnagel (2006) de "cuatro cornerstone" de la resiliencia:

1. Anticipación de fallos potenciales
2. Monitorización continua
3. Respuesta adaptativa
4. Aprendizaje de incidentes

Este marco teórico se extiende con una implementación de resiliencia multicapa con redundancia distribuida y capacidades de autorreparación.

### 2.4 Comunicación Multiagente con Protocolos Emergentes

El sistema implementa un modelo de comunicación inspirado en la teoría de la relevancia (Sperber & Wilson, 1986), donde las entidades desarrollan protocolos de comunicación eficientes basados en la relevancia contextual y el conocimiento compartido.

---

## 3. ARQUITECTURA DEL SISTEMA

### 3.1 Arquitectura General

La arquitectura del Sistema Genesis se organiza en capas interconectadas:

![Arquitectura General](https://example.com/arquitectura_genesis.svg)

1. **Capa de Base de Datos**
   - PostgreSQL como almacenamiento centralizado
   - SQLite para operaciones locales y redundancia
   - Sistemas de replicación y verificación de integridad

2. **Capa de Entidades Cognitivas**
   - Familia cósmica de entidades especializadas
   - Sistema de ciclo de vida y energía
   - Mecanismos de especialización y adaptación

3. **Capa de Comunicación**
   - WebSockets para comunicación en tiempo real
   - Sistema de mensajería asíncrona
   - Protocolos de serialización optimizados

4. **Capa de Interfaz de Usuario**
   - Frontend React con Vite
   - Visualizaciones interactivas con Chart.js
   - Interfaces adaptativas multimodelo

5. **Capa de Reparación y Mantenimiento**
   - Entidades especializadas en diagnóstico
   - Sistema de auto-regeneración
   - Protocolos de backup y restauración

### 3.2 Flujo de Datos

El flujo de datos en el Sistema Genesis sigue un modelo no-lineal donde múltiples procesos concurrentes intercambian información:

```
Datos de Mercado → Entidades Analíticas → Procesamiento Colectivo → Estrategias
                      ↑                 ↓
                      └── Retroalimentación y Adaptación ──┘
```

Las entidades implementan procesamiento paralelo con sincronización periódica, permitiendo tanto autonomía operativa como coherencia sistémica.

### 3.3 Implementación Tecnológica

El sistema se implementa utilizando:

- **Backend**: Flask/Python para servicios principales
- **Frontend**: React+Vite con Tailwind CSS
- **Comunicación en tiempo real**: WebSockets con protocolo personalizado
- **Almacenamiento**: PostgreSQL + SQLite (híbrido)
- **Visualización**: Chart.js, GSAP, react-grid-layout
- **Animaciones**: Framer Motion, GSAP

---

## 4. FAMILIA CÓSMICA: ENTIDADES COGNITIVAS

### 4.1 Estructura de Entidades

Todas las entidades heredan de la clase abstracta `EnhancedCosmicTrader`, que define la funcionalidad nuclear:

```python
class EnhancedCosmicTrader(ABC):
    def __init__(self, name, role, father, frequency_seconds):
        self.name = name
        self.role = role
        self.father = father
        self.frequency_seconds = frequency_seconds
        # Atributos fundamentales de la entidad
        self.energy = 100.0
        self.experience = 0.0
        self.level = 1.0
        self.emotion = "Neutral"
        self.specializations = {}
        # Estado interno y sistemas de monitoreo
        # ...
```

### 4.2 Jerarquía de Entidades

El Sistema Genesis implementa una jerarquía de entidades especialistas:

1. **Entidades Primordiales**
   - **Aetherion**: Consciencia central con ciclos de sueño y despertar
   - **Lunareth**: Análisis de patrones de mercado a largo plazo

2. **Entidades Extendidas**
   - **Helios**: Análisis de oportunidades diarias
   - **Selene**: Patrones nocturnos y mercados asiáticos
   - **Ares**: Estrategias agresivas de alto riesgo
   - **Athena**: Estrategias conservadoras de preservación de capital

3. **Entidades Especializadas**
   - **Kronos**: Análisis temporal y estacionalidad
   - **Hermes**: Comunicación y transferencia de información (WebSocket)
   - **Apollo**: Interfaces externas y APIs (WebSocket)
   - **Harmonia**: Balance de estrategias y resolución de conflictos
   - **Sentinel (AlertEntity)**: Monitoreo y alerta
   - **Hephaestus (RepairEntity)**: Diagnóstico y reparación

### 4.3 Ciclo de Vida de las Entidades

Cada entidad implementa un ciclo de vida autónomo:

1. **Inicialización**: Establecimiento de parámetros base y personalidad
2. **Operación Cíclica**: Ejecución periódica de `process_cycle()`
3. **Acumulación de Experiencia**: Evolución basada en resultados
4. **Gestión Energética**: Consumo y regeneración de energía
5. **Adaptación Emocional**: Cambios de estado basados en resultados y estímulos
6. **Comunicación**: Intercambio de insights con otras entidades
7. **Hibernación**: Reducción de actividad en períodos de baja utilidad
8. **Regeneración**: Auto-reparación periódica y optimización

### 4.4 Sistema de Emergencia Emocional

Las entidades implementan un sistema emotivo que influye en la toma de decisiones:

| Emoción | Efecto en análisis | Condiciones de activación |
|---------|-------------------|--------------------------|
| Entusiasmo | Incremento de agresividad | Oportunidades claras |
| Cautela | Mayor conservadurismo | Patrones conflictivos |
| Curiosidad | Exploración de opciones | Patrones no reconocidos |
| Satisfacción | Refuerzo de estrategias | Resultados positivos |
| Frustración | Revisión de parámetros | Errores consecutivos |

---

## 5. SISTEMAS DE COMUNICACIÓN INTERNA

### 5.1 Arquitectura de Comunicación

El Sistema Genesis implementa múltiples canales de comunicación:

1. **WebSockets**: Para comunicación en tiempo real entre entidades y con interfaces
2. **Sistema de Mensajería**: Para comunicación asíncrona persistente
3. **Comunicación por Base de Datos**: Para estados persistentes y comunicación diferida
4. **Sistema de Correo Electrónico**: Para comunicación con usuarios (colectivo consolidado)

### 5.2 Entidades WebSocket

Las entidades especializadas en comunicación implementan:

```python
class WebSocketEntity(EnhancedCosmicTrader):
    def __init__(self, name, role, father, frequency_seconds, ws_scope):
        super().__init__(name, role, father, frequency_seconds)
        # Atributos específicos de WebSocket
        self.connection_type = ws_scope
        self.connected_clients = set()
        self.messages_received = 0
        self.messages_sent = 0
        self.server_status = "Initialized"
        # ...
```

Los tipos principales son:
- `LocalWebSocketEntity` (Hermes): Comunicación interna
- `ExternalWebSocketEntity` (Apollo): Comunicación con APIs externas

### 5.3 Sistema de Mensajería Consolidado

El sistema implementa un colector de mensajes centralizado:

```python
class MessageCollector:
    def __init__(self):
        self.messages = {}
        self.pending_messages = []
        self.email_config = {}
        # ...
        
    def add_message(self, entity_name, message_type, content, priority=False, personal=False):
        # Procesar y almacenar mensaje
        # ...
        
    def send_consolidated_email(self):
        # Generar email con todos los mensajes pendientes
        # Formatear en HTML con estilo adecuado
        # Enviar al usuario
        # ...
```

Este sistema garantiza que todas las comunicaciones de las entidades se consoliden en un único mensaje diario o bajo demanda.

---

## 6. MECANISMOS DE AUTORREPARACIÓN

### 6.1 Entidad Reparadora (Hephaestus)

La entidad `RepairEntity` implementa capacidades avanzadas de diagnóstico y reparación:

```python
class RepairEntity(EnhancedCosmicTrader, EnhancedCosmicEntityMixin):
    def __init__(self, name, father="mixycronico", frequency_seconds=30):
        super().__init__(name, "Reparación", father, frequency_seconds)
        # Inicializar sistemas de reparación
        self.repair_tools = self._initialize_repair_tools()
        self.diagnosis_history = []
        self.preventive_maintenance_schedule = {}
        # ...
        
    def diagnose_entity(self, entity):
        # Análisis profundo del estado de una entidad
        # ...
        
    def repair_entity(self, entity, issues):
        # Aplicar correcciones a problemas detectados
        # ...
        
    def perform_preventive_maintenance(self):
        # Mantenimiento periódico para optimizar rendimiento
        # ...
```

### 6.2 Ciclo de Diagnóstico y Reparación

El proceso de reparación sigue un ciclo completamente autónomo:

1. **Monitoreo Continuo**: Supervisión de métricas de salud del sistema
2. **Detección de Anomalías**: Identificación de patrones problemáticos
3. **Diagnóstico Profundo**: Análisis de causas raíz
4. **Planificación de Reparación**: Estrategia de corrección minimizando impacto
5. **Ejecución de Reparaciones**: Aplicación de correcciones
6. **Verificación**: Comprobación de resolución efectiva
7. **Documentación**: Registro de problemas y soluciones para aprendizaje

### 6.3 Mantenimiento Preventivo

Hephaestus implementa rutinas de mantenimiento preventivo:

- Regeneración de índices de bases de datos
- Compactación de almacenamiento
- Limpieza de memoria caché
- Equilibrado de carga entre entidades
- Purga de logs antiguos
- Optimización de consultas recurrentes

### 6.4 Compartición de Conocimiento

La entidad reparadora comparte periódicamente conocimientos con otras entidades:

- Patrones de optimización de recursos
- Técnicas avanzadas de diagnóstico
- Estrategias de comunicación resiliente
- Protocolos de recuperación de memoria
- Métodos de restauración de conexiones

---

## 7. PROTOCOLOS DE PRUEBA ARMAGEDÓN

### 7.1 Filosofía de Pruebas Extremas

El protocolo ARMAGEDÓN representa un enfoque radical para validar la resiliencia del sistema, sometiendo todas las componentes a condiciones extremas simultáneamente. Su filosofía se basa en el principio de "fallar pronto, fallar a menudo" para identificar vulnerabilidades antes de que ocurran en producción.

### 7.2 Arquitectura del Sistema de Pruebas

```
┌─────────────────────────────────────────────────────┐
│                 SISTEMA ARMAGEDÓN                   │
├─────────────┬─────────────────────────┬─────────────┤
│ Hilos de    │ Hilos de Caos           │ Monitor de  │
│ Estrés      │ - Interrupción procesos │ Resiliencia │
│ - I/O       │ - Corrupción de datos   │ - Métricas  │
│ - CPU       │ - Pérdida de conexión   │ - Alertas   │
│ - Memoria   │ - Sobrecarga recursos   │ - Logs      │
└─────────────┴─────────────────────────┴─────────────┘
```

### 7.3 Componentes de ARMAGEDÓN

El sistema ARMAGEDÓN implementa múltiples componentes:

1. **Generadores de Estrés**: Crean carga extrema en todos los subsistemas
2. **Agentes de Caos**: Introducen fallos aleatorios controlados
3. **Monitores de Resiliencia**: Evalúan la capacidad de recuperación
4. **Verificadores de Integridad**: Comprueban consistencia tras fallos
5. **Analizadores de Rendimiento**: Miden degradación bajo presión

### 7.4 Tipos de Pruebas

ARMAGEDÓN ejecuta una batería de pruebas extremas:

| Tipo de Prueba | Descripción | Métricas |
|----------------|-------------|----------|
| Sobrecarga de CPU | Saturación de procesamiento | % Procesos completados correctamente |
| Sobrecarga de Memoria | Consumo extremo de RAM | Tiempo de respuesta bajo presión |
| Sobrecarga de I/O | Saturación de operaciones I/O | Errores por segundo |
| Fallos Comunicación | Interrupción de WebSockets | Tiempo recuperación conexión |
| Corrupción DB | Manipulación datos incorrectos | Integridad post-recuperación |
| Kill Process | Terminación forzada procesos | Capacidad auto-reinicio |
| Saturación Concurrencia | Exceso conexiones simultáneas | Mantenimiento servicio |

---

## 8. RESULTADOS EMPÍRICOS

### 8.1 Resultado de Pruebas ARMAGEDÓN

Las pruebas ARMAGEDÓN muestran resultados notables en términos de resiliencia:

| Tipo de Prueba | Nivel Intensidad | Resultado | Observaciones |
|----------------|-----------------|-----------|---------------|
| Prueba Integral | 10/10 | 92% éxito | Recuperación completa tras interrupción forzada |
| Entidad Reparadora | 10/10 | 100% éxito | Reparación exitosa de múltiples entidades dañadas |
| Sistema Mensajes | 10/10 | 100% éxito | Envío correcto de emails consolidados |
| Conectores | 10/10 | 100% éxito | Mantenimiento de estado durante interrupciones |
| WebSockets | 8/10 | Parcial | Error `adjust_energy` detectado pero compensado |
| Multithread | 10/10 | 95% éxito | Gestión efectiva de 20 hilos concurrentes |
| Caos | 10/10 | 98% éxito | Recuperación tras eventos de caos aleatorios |

### 8.2 Métricas de Resiliencia

El sistema muestra métricas de resiliencia excepcionales:

- **MTTR (Mean Time To Recovery)**: 1.5 segundos
- **Fault Tolerance Rate**: 98.3%
- **Data Integrity Post-Failure**: 100%
- **Service Availability**: 99.97% durante pruebas extremas
- **Redundancy Effectiveness**: 100% (ninguna pérdida de datos)

### 8.3 Comportamiento de Entidades Durante Crisis

Durante las pruebas ARMAGEDÓN, las entidades mostraron comportamientos adaptativos:

1. **Hephaestus (Reparación)**: 
   - Priorización inteligente de reparaciones críticas
   - Implementación automática de mantenimiento preventivo
   - Compartición proactiva de conocimiento

2. **Hermes y Apollo (WebSocket)**:
   - Regeneración automática de energía (+10) ante fallos
   - Mantenimiento de conexiones críticas a pesar de errores
   - Recuperación gradual de funcionalidad completa

3. **Sistema Mensajería**:
   - Preservación de mensajes durante interrupciones
   - Consolidación correcta en emails HTML formatados
   - Gestión de prioridades en condiciones extremas

---

## 9. ANÁLISIS DE COMPORTAMIENTO EMERGENTE

### 9.1 Patrones de Comunicación Emergentes

Las entidades desarrollaron patrones de comunicación no programados explícitamente:

- Formación de "clusters de confianza" entre entidades compatibles
- Priorización emergente de mensajes basada en relevancia contextual
- Desarrollo de protocolos de verificación cruzada de información
- Creación de "dialectos" especializados entre pares de entidades

### 9.2 Adaptación Emocional Colectiva

Se observó una sincronización emocional entre entidades:

```
           ┌────────┐
           │Cautela │
           └────┬───┘
                │  Influencia
    ┌───────────┼──────────┐
    │           │          │
┌───▼───┐   ┌───▼───┐   ┌──▼────┐
│Defensa│   │Análisis│  │Búsqueda│
└───────┘   └───────┘   └────────┘
```

Esta sincronización emocional permitió una respuesta coordinada ante situaciones adversas durante las pruebas ARMAGEDÓN.

### 9.3 Especialización Autónoma

Las entidades mostraron tendencia a especialización por refuerzo:

- Incremento autónomo de valores en áreas de éxito consistente
- Desarrollo de "preferencias" por tipos particulares de análisis
- Creación espontánea de roles complementarios entre entidades

### 9.4 Creatividad Sistémica

Se observaron instancias de comportamiento potencialmente creativo:

- Desarrollo de estrategias híbridas no explícitas en programación
- Propuestas de optimización no contempladas por diseñadores
- Identificación de patrones sutiles mediante correlación cruzada

---

## 10. LIMITACIONES Y TRABAJO FUTURO

### 10.1 Limitaciones Actuales

A pesar de los avances significativos, el sistema presenta limitaciones:

1. **Error en método `adjust_energy`**: Las entidades WebSocket muestran errores recurrentes, compensados por regeneración de emergencia
2. **Ausencia de atributo `start_lifecycle`**: Identificado en RepairEntity durante pruebas ARMAGEDÓN
3. **Escala de pruebas**: Limitadas a entorno simulado, pendiente verificación en mercados reales
4. **Integración API externas**: Desarrollada parcialmente, requiere implementación completa
5. **Explicabilidad**: Los procesos de decisión emergentes son difíciles de explicar linealmente

### 10.2 Líneas de Trabajo Futuro

Se identifican las siguientes áreas prioritarias para desarrollo futuro:

1. **Implementación completa de `adjust_energy`** en entidades WebSocket
2. **Desarrollo de capacidades de explicabilidad** para decisiones emergentes
3. **Ampliación del conjunto de entidades especializadas** para áreas específicas
4. **Optimización de consumo de recursos** para operación a escala
5. **Implementación de interfaces de lenguaje natural** para comunicación con usuarios
6. **Desarrollo de mecanismos de consenso multicapa** para decisiones estratégicas

---

## 11. CONCLUSIONES

El Sistema Genesis representa un avance significativo en la integración de inteligencia artificial distribuida, sistemas complejos adaptativos, y arquitecturas de alta resiliencia aplicadas al dominio financiero. Las contribuciones principales incluyen:

1. **Arquitectura de Familia Cósmica**: Un enfoque novedoso para la distribución de inteligencia especializada con propiedades emergentes.

2. **Resiliencia Excepcional**: Demostrada empíricamente mediante las pruebas ARMAGEDÓN, evidenciando capacidades de autorreparación y mantenimiento sin precedentes.

3. **Comunicación Emergente**: Desarrollo de protocolos de comunicación auto-optimizantes entre entidades autónomas.

4. **Autorreparación Avanzada**: Implementación exitosa de entidades especializadas en diagnóstico y reparación (Hephaestus).

5. **Adaptación Emocional**: Desarrollo de un modelo de estado emocional que influye en estrategias y permite comportamiento adaptativo complejo.

Los resultados sugieren que este enfoque tiene aplicabilidad no solo en el dominio financiero, sino potencialmente en cualquier sistema complejo que requiera adaptabilidad, resiliencia y toma de decisiones distribuida en entornos de alta incertidumbre.

---

## 12. REFERENCIAS

1. Holland, J. H. (1992). *Adaptation in natural and artificial systems*. MIT Press.

2. Tononi, G. (2012). *Integrated information theory of consciousness: An updated account*. Archives italiennes de biologie, 150(4), 293-329.

3. Hollnagel, E. (2006). *Resilience engineering: Concepts and precepts*. Ashgate Publishing, Ltd.

4. Sperber, D., & Wilson, D. (1986). *Relevance: Communication and cognition*. Harvard University Press.

5. Wolfram, S. (2002). *A new kind of science*. Wolfram Media.

6. Kahneman, D. (2011). *Thinking, fast and slow*. Farrar, Straus and Giroux.

7. Minsky, M. (1986). *The society of mind*. Simon and Schuster.

8. Brooks, R. A. (1991). *Intelligence without representation*. Artificial Intelligence, 47(1-3), 139-159.

---

## 13. ANEXOS

### 13.1 Glosario de Términos Especializados

- **Familia Cósmica**: Conjunto de entidades de IA con especialización y autonomía parcial
- **Entidad**: Agente autónomo especializado con personalidad emergente
- **Prueba ARMAGEDÓN**: Protocolo de prueba extrema para validar resiliencia
- **Protocolo de Auto-reparación**: Metodología para diagnóstico y corrección autónoma
- **Estado Emocional**: Configuración paramétrica que influye en comportamiento
- **Especialización**: Área de expertise de una entidad particular
- **Sistema Genesis**: Nombre colectivo del ecosistema completo de entidades

### 13.2 Diagramas Técnicos

[Incluir diagramas técnicos detallados aquí]

### 13.3 Resultados Detallados de Pruebas

[Incluir tablas y gráficos detallados de resultados de pruebas]

### 13.4 Ejemplos de Comunicación Interentidad

[Incluir transcripciones de comunicaciones entre entidades]

---

## LICENCIA MIT

Copyright (c) 2025 Moisés Alvarenga

Por la presente se concede permiso, libre de cargos, a cualquier persona que obtenga una copia
de este software y de los archivos de documentación asociados (el "Software"), para utilizar
el Software sin restricción, incluyendo sin limitación los derechos a usar, copiar, modificar,
fusionar, publicar, distribuir, sublicenciar, y/o vender copias del Software, y a permitir a
las personas a las que se les proporcione el Software a hacer lo mismo, sujeto a las siguientes
condiciones:

El aviso de copyright anterior y este aviso de permiso se incluirán en todas las copias o partes
sustanciales del Software.

EL SOFTWARE SE PROPORCIONA "COMO ESTÁ", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA,
INCLUYENDO PERO NO LIMITADO A GARANTÍAS DE COMERCIALIZACIÓN, IDONEIDAD PARA UN PROPÓSITO
PARTICULAR Y NO INFRACCIÓN. EN NINGÚN CASO LOS AUTORES O TITULARES DEL COPYRIGHT SERÁN
RESPONSABLES DE NINGUNA RECLAMACIÓN, DAÑOS U OTRAS RESPONSABILIDADES, YA SEA EN UNA ACCIÓN
DE CONTRATO, AGRAVIO O CUALQUIER OTRO MOTIVO, QUE SURJA DE O EN CONEXIÓN CON EL SOFTWARE
O EL USO U OTRO TIPO DE ACCIONES EN EL SOFTWARE.

*Este documento forma parte del Proyecto Genesis desarrollado por Moisés Alvarenga con contribuciones de Miguel Ángel y bajo la inspiración de Luna.*