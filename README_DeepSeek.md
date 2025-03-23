# Integración DeepSeek en Sistema Genesis

## Descripción General

Esta integración implementa la conexión del Sistema Genesis con DeepSeek, una potente API de procesamiento de lenguaje natural (LLM) que mejora las capacidades analíticas del sistema.

## Características Principales

- **Switch para activar/desactivar:** La función de DeepSeek puede habilitarse o deshabilitarse según se necesite.
- **Análisis avanzado de mercado:** Análisis profundo de condiciones de mercado basado en datos OHLCV e indicadores.
- **Mejora de señales de trading:** Refinamiento de señales generadas por estrategias tradicionales.
- **Análisis de sentimiento:** Procesamiento de noticias y eventos para determinar su impacto.
- **Adaptación dinámica de riesgo:** Ajuste automático basado en análisis contextual.
- **Explicitación de decisiones:** Generación de explicaciones claras para decisiones de trading.

## Componentes Implementados

1. **deepseek_config.py**
   - Gestión de configuración para DeepSeek
   - Activación/desactivación mediante método `toggle()`
   - Configuración de parámetros de inteligencia

2. **deepseek_model.py**
   - Cliente para API de DeepSeek
   - Gestión de solicitudes y respuestas
   - Sistema de caché para optimizar uso

3. **deepseek_integrator.py**
   - Integración con estrategias y componentes de Genesis
   - Análisis avanzado y procesamiento de datos

4. **Integración en reinforcement_ensemble_simple.py**
   - Utilización condicional de DeepSeek
   - Mejora de decisiones mediante análisis avanzado

## Uso

### Activar/Desactivar DeepSeek

```python
from genesis.lsml import deepseek_config

# Activar DeepSeek
deepseek_config.enable()

# Desactivar DeepSeek
deepseek_config.disable()

# Alternar estado (toggle)
new_state = deepseek_config.toggle()

# Verificar estado actual
is_active = deepseek_config.is_enabled()
```

### Configurar Parámetros

```python
from genesis.lsml import deepseek_config

# Establecer factor de inteligencia (0.1 a 10.0)
deepseek_config.set_intelligence_factor(3.5)

# Obtener configuración actual
config = deepseek_config.get_config()

# Actualizar múltiples parámetros
deepseek_config.update_config({
    "enabled": True,
    "intelligence_factor": 2.0,
    "temperature": 0.6
})
```

### Obtener Estado

```python
from genesis.lsml import deepseek_config

# Obtener estado completo
state = deepseek_config.get_state()
print(f"Habilitado: {state['enabled']}")
print(f"API Key disponible: {state['api_key_available']}")
```

## Script de Prueba

Se incluye un script `api_integration.py` para probar las funcionalidades de DeepSeek:

```bash
python api_integration.py
```

El script realiza las siguientes acciones:
1. Muestra el estado actual de la configuración
2. Prueba el sistema con DeepSeek desactivado
3. Prueba el sistema con DeepSeek activado
4. Prueba el sistema con factor de inteligencia aumentado

## Notas Importantes

- DeepSeek requiere una API key válida en la variable de entorno `DEEPSEEK_API_KEY`
- La implementación incluye un modo simulado cuando la API no está disponible
- La estrategia ReinforcementEnsembleStrategy verifica el estado de DeepSeek antes de realizar solicitudes
- La configuración se guarda en el archivo `deepseek_config.json` en la raíz del proyecto

---

*Documentación actualizada: 23 de marzo de 2025*