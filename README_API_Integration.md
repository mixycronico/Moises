# Sistema de Integración de APIs para Genesis

## Descripción

Este módulo proporciona una interfaz unificada para interactuar con múltiples APIs externas, permitiendo al Sistema Genesis incorporar datos e inteligencia de diversas fuentes para enriquecer sus capacidades de análisis y toma de decisiones.

## APIs Soportadas

1. **DeepSeek**
   - Análisis avanzado con IA de texto
   - Mejora de señales de trading mediante interpretación de contexto
   - Procesamiento de noticias y eventos

2. **Alpha Vantage**
   - Datos históricos de precios
   - Indicadores técnicos
   - Datos fundamentales

3. **NewsAPI**
   - Noticias y eventos financieros
   - Análisis de sentimiento de mercado
   - Seguimiento de tendencias

4. **CoinMarketCap**
   - Información detallada de mercado
   - Datos de capitalización
   - Métricas de criptomonedas

5. **Reddit**
   - Análisis de sentimiento social
   - Tendencias en comunidades de trading
   - Alertas tempranas de movimientos de mercado

## Características Principales

- **Gestión unificada de claves API**: Todas las claves se gestionan desde variables de entorno
- **Control de límites de tasa**: Prevención automática de excesos de solicitudes
- **Reintentos inteligentes**: Backoff exponencial en caso de fallos
- **Caché de resultados**: Reducción de llamadas duplicadas
- **Sistema de activación/desactivación**: Control preciso de qué APIs utilizar

## Configuración

### Variables de Entorno

Para utilizar las APIs, es necesario configurar las siguientes variables de entorno:

```
DEEPSEEK_API_KEY=sk_...
ALPHA_VANTAGE_API_KEY=...
NEWS_API_KEY=...
COINMARKETCAP_API_KEY=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
```

### Verificación de Estado

Para verificar qué APIs están disponibles:

```bash
python api_integration.py --status
```

## Uso

### Desde Línea de Comandos

```bash
# Ver estado de todas las APIs configuradas
python api_integration.py --status

# Probar todas las APIs configuradas
python api_integration.py --test

# Probar solo la integración con DeepSeek
python api_integration.py --deepseek

# Ejecutar todas las pruebas
python api_integration.py --all
```

### Desde Código

```python
from genesis.api_integration import api_manager

# Inicializar el gestor de APIs
await api_manager.initialize()

# Consultar Alpha Vantage
alpha_data = await api_manager.get_alpha_vantage_data(
    function="TIME_SERIES_DAILY", 
    symbol="BTC",
    outputsize="compact"
)

# Obtener noticias recientes sobre Bitcoin
news = await api_manager.get_news_api_data(
    query="bitcoin", 
    language="es", 
    sortBy="publishedAt"
)

# Consultar información de mercado
market_data = await api_manager.get_coinmarketcap_data(
    endpoint_path="cryptocurrency/listings/latest", 
    limit=10
)

# Obtener análisis avanzado con DeepSeek
deepseek_result = await api_manager.call_deepseek_api(
    messages=[
        {"role": "system", "content": "Analiza las condiciones de mercado de Bitcoin."},
        {"role": "user", "content": "¿Es buen momento para comprar Bitcoin?"}
    ]
)

# No olvidar cerrar las conexiones al finalizar
await api_manager.close()
```

## Integración con Estrategias de Trading

Las APIs externas se pueden integrar con las estrategias de trading del Sistema Genesis para obtener información adicional:

```python
# Ejemplo en una estrategia
from genesis.api_integration import api_manager

class EnhancedStrategy(BaseStrategy):
    async def analyze_market(self, symbol):
        # Obtener datos básicos
        basic_data = await self._get_market_data(symbol)
        
        # Enriquecer con datos de APIs externas
        news = await api_manager.get_news_api_data(
            query=symbol.split('/')[0],  # Ej: BTC de BTC/USDT
            pageSize=5
        )
        
        sentiment = await self._analyze_sentiment(news)
        
        # Combinar todos los datos para una decisión más informada
        return self._make_decision(basic_data, sentiment)
```

## Gestión de Errores y Límites

El sistema gestiona automáticamente:

1. **Reintentos inteligentes**: En caso de error temporal (5xx, timeout)
2. **Control de límites de tasa**: Espera automática si se acerca al límite
3. **Fallback**: Continúa operando con datos internos si una API no está disponible

## Ampliación

Para añadir soporte para una nueva API:

1. Añadir clave en environment variable
2. Actualizar `_load_api_keys()` en `api_manager.py`
3. Implementar método específico similar a `get_alpha_vantage_data()`
4. Actualizar límites de tasa en `APIRateLimiter`