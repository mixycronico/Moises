# Genesis - Sistema Avanzado de Trading Algorítmico

Sistema profesional para trading algorítmico de criptomonedas con enfoque en resiliencia, prevención de fallos en cascada y arquitectura modular de alto rendimiento.

## Características Principales

- **Motor de Eventos Optimizado**: Procesamiento no bloqueante con manejo avanzado de eventos para máxima eficiencia.
- **Prevención de Fallos en Cascada**: Implementación del patrón Circuit Breaker y aislamiento de componentes.
- **Monitor de Componentes**: Detección automática de componentes problemáticos y recuperación inteligente.
- **Estrategias Avanzadas**: Indicadores técnicos, análisis de sentimiento y estrategias adaptativas.
- **Backtesting de Alto Rendimiento**: Simulación de estrategias con datos históricos reales.
- **Gestión de Riesgos**: Posicionamiento inteligente, stop-loss dinámico y trailing-stop.
- **Notificaciones en Tiempo Real**: Alertas configurables vía email, SMS y webhooks.
- **Panel de Control Interactivo**: Visualización en tiempo real del rendimiento del sistema.

## Arquitectura

Genesis utiliza una arquitectura basada en componentes independientes que se comunican a través de un bus de eventos:

- **Core**: Motor de eventos, gestión de componentes y bus de comunicación.
- **Exchange**: Integración con exchanges de criptomonedas (Binance, Kraken, etc).
- **Data**: Gestión y procesamiento de datos de mercado.
- **Strategy**: Implementación de estrategias de trading.
- **Execution**: Ejecución y seguimiento de órdenes.
- **Risk**: Gestión de riesgos y capital.
- **Analytics**: Análisis de rendimiento y reporting.
- **Notification**: Sistema de alertas y notificaciones.

## Prevención de Fallos en Cascada

El sistema implementa varios mecanismos para prevenir fallos en cascada:

1. **ComponentMonitor**: Monitorea continuamente la salud de cada componente y aísla aquellos que:
   - No responden dentro de un tiempo máximo
   - Fallan consistentemente
   - Reportan explícitamente estado no saludable

2. **Circuit Breaker Pattern**: Implementación completa del patrón Circuit Breaker con estados:
   - **Closed**: Operación normal, los eventos fluyen normalmente
   - **Open**: Componente aislado, los eventos no llegan al componente
   - **Half-Open**: Fase de recuperación, se permite un número limitado de eventos para verificar recuperación

3. **Componentes con Conciencia de Dependencias**: Cada componente conoce sus dependencias y:
   - Actualiza su estado basado en el estado de sus dependencias
   - Implementa recuperación automática cuando las dependencias se recuperan
   - Maneja adecuadamente los cambios de estado de las dependencias

4. **Gestión Robusta de Operaciones Asíncronas**:
   - Todas las operaciones asíncronas tienen timeouts configurables
   - Reintentos automáticos con backoff exponencial
   - Manejo defensivo de respuestas potencialmente nulas o incompletas

## Instrucciones de Uso

### Requisitos

- Python 3.10+
- PostgreSQL (para almacenamiento de datos)
- Cuenta en exchanges de criptomonedas (Binance Testnet para pruebas)

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/usuario/genesis.git
cd genesis

# Configurar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar base de datos
python db_setup.py
```

### Configuración

Editar el archivo `.env` con tus credenciales:

```
# Configuración de la base de datos
DATABASE_URL=postgresql://usuario:password@localhost/genesis

# Credenciales de exchanges (opcional para modo simulación)
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret

# Configuración de notificaciones (opcional)
EMAIL_SERVER=smtp.example.com
EMAIL_USER=usuario@example.com
EMAIL_PASSWORD=tu_password
```

### Ejecución

```bash
# Iniciar la aplicación web
python main.py

# Ejecutar en modo testnet
python main.py --testnet

# Modo backtest
python main.py --backtest --strategy rsi --start-date 2023-01-01 --end-date 2023-12-31
```

### Ejecución de Pruebas

```bash
# Ejecutar todas las pruebas
pytest

# Ejecutar pruebas específicas
pytest tests/unit/core/

# Ejecutar pruebas de prevención de fallos en cascada
python scripts/execute_all_cascade_tests.py
```

## Documentación

La documentación completa está disponible en los siguientes archivos:

- `docs/prevencion_fallos_cascada.md`: Documentación técnica sobre la prevención de fallos en cascada
- `docs/reporte_monitor_componentes.md`: Informe sobre el rendimiento del monitor de componentes
- `docs/reporte_prevencion_fallos_cascada.md`: Informe completo de pruebas de prevención de fallos

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.

## Contribuir

Las contribuciones son bienvenidas. Por favor, sigue estas pautas:

1. Haz fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/amazing-feature`)
3. Realiza tus cambios
4. Ejecuta las pruebas (`pytest`)
5. Haz commit de tus cambios (`git commit -m 'Add amazing feature'`)
6. Haz push a la rama (`git push origin feature/amazing-feature`)
7. Abre un Pull Request