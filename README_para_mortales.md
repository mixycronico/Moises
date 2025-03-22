# Sistema Genesis

*Una plataforma robusta de procesamiento de eventos para trading de criptomonedas*

## Descripción General

El Sistema Genesis es una plataforma híbrida para el procesamiento resiliente de eventos en trading de criptomonedas, diseñada para mantener su operación incluso bajo condiciones extremas como fallos de componentes, latencia alta o sobrecarga de eventos.

## Características Principales

- Arquitectura de microservicios con comunicación asíncrona
- Múltiples capas de resiliencia (reintentos, circuit breakers, checkpoints)
- Procesamiento distribuido y paralelo de eventos
- Adaptación automática a condiciones cambiantes
- Monitoreo exhaustivo y auto-recuperación

## Requisitos

- Python 3.8+
- PostgreSQL 12+
- 2GB RAM (mínimo recomendado)
- Conexión estable a Internet

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/genesis-system.git
   cd genesis-system
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Configurar la base de datos:
   ```bash
   # Crear la base de datos PostgreSQL
   createdb genesis
   
   # Configurar las tablas
   python db_setup.py
   ```

4. Configurar variables de entorno (crear archivo `.env`):
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/genesis
   API_PORT=5000
   WS_PORT=5001
   LOG_LEVEL=INFO
   ```

## Uso Básico

### Iniciar el sistema

```bash
python main.py
```

### Ejecutar tests

```bash
pytest tests/
```

### Monitoreo

```bash
# Ver logs en tiempo real
tail -f logs/app.log

# Ver métricas del sistema
python scripts/metrics.py
```

## Arquitectura

El sistema se compone de varios módulos interconectados:

- **Coordinador central**: Gestiona la comunicación entre componentes
- **Bus de eventos**: Distribuye eventos asíncronos en el sistema
- **Componentes modulares**: Encapsulan funcionalidades específicas
- **Mecanismos de resiliencia**: Garantizan operación continua

## Componentes Principales

- `Coordinator`: Gestiona la comunicación y orquestación
- `ComponentAPI`: Interfaz base para todos los componentes
- `CircuitBreaker`: Aísla componentes fallidos
- `Checkpointer`: Guarda y restaura estados
- `RetryHandler`: Reintenta operaciones fallidas

## Ejemplo de Código

```python
# Registrar un componente en el sistema
coordinator = Coordinator()
strategy = StrategyComponent("macd_crossover")
coordinator.register_component("strategy1", strategy)

# Enviar una solicitud
result = await coordinator.request(
    "strategy1", 
    "execute", 
    {"symbol": "BTC/USDT", "timeframe": "1h"},
    "main"
)

# Emitir un evento
await coordinator.emit_event(
    "trade_executed",
    {"symbol": "BTC/USDT", "side": "buy", "amount": 0.1, "price": 45000},
    "exchange"
)
```

## Estructura de Directorios

```
genesis/
├── core/              # Funcionalidades principales
├── components/        # Componentes del sistema
├── api/               # APIs REST y WebSocket
├── db/                # Modelos y conexión a DB
├── utils/             # Utilidades comunes
├── scripts/           # Scripts auxiliares
├── tests/             # Tests automatizados
├── logs/              # Archivos de logs
├── checkpoints/       # Estados guardados
├── docs/              # Documentación
├── main.py            # Punto de entrada principal
├── requirements.txt   # Dependencias
└── README.md          # Este archivo
```

## Modos de Operación

El sistema puede funcionar en diferentes modos:

- **Normal**: Funcionalidad completa
- **Seguro**: Funcionalidades reducidas (durante problemas)
- **Emergencia**: Solo funciones críticas (fallos severos)

## Resolución de Problemas

| Problema | Solución |
|----------|----------|
| Error de conexión a DB | Verificar credenciales y disponibilidad de PostgreSQL |
| Componente bloqueado | Reiniciar el componente: `python scripts/restart.py <component_id>` |
| Alto uso de CPU | Verificar sobrecarga de eventos, considerar escalado |
| Errores de timeout | Aumentar valores de timeout en config.py |

## Contribuir

1. Fork el repositorio
2. Crear una rama para tu característica (`git checkout -b feature/amazing-feature`)
3. Commit tus cambios (`git commit -m 'Add some amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está licenciado bajo [MIT License](LICENSE).

## Contacto

Equipo de Desarrollo - dev@example.com

---

*Nota: Esta es la documentación simplificada para usuarios regulares. Para acceder a la documentación técnica avanzada sobre los mecanismos de resiliencia especializados, contactar al equipo de desarrollo.*