# Guía de Pruebas del Módulo de Backtesting

## Introducción

Este documento proporciona instrucciones detalladas para probar correctamente el módulo de backtesting del sistema Genesis. El backtesting es un componente crítico que permite evaluar las estrategias de trading en datos históricos antes de implementarlas en un entorno real.

## Problemas Comunes y Soluciones

### 1. Problema: Datos insuficientes para estrategias

**Síntoma**: Las estrategias reciben solo un punto de datos para generar señales, lo que hace imposible calcular indicadores que requieren datos históricos.

**Solución**: El método `run_strategy` en `BacktestEngine` debe proporcionar una ventana de datos históricos suficiente para cada punto evaluado.

```python
# Corrección aplicada al método run_strategy
# En lugar de pasar solo la fila actual:
row_df = pd.DataFrame([data_point], index=[idx])

# Proporcionar un subconjunto con suficientes datos históricos:
window_size = max(30, i + 1)  # Por defecto 30 barras o todas hasta ahora
start_idx = max(0, end_idx - window_size)
history_df = df.iloc[start_idx:end_idx].copy()
```

### 2. Problema: Manejo inadecuado de señales

**Síntoma**: Las señales generadas por la estrategia no se procesan correctamente en la simulación de trading.

**Solución**: Asegurarse de que el formato de las señales sea coherente y se maneje correctamente en `simulate_trading_with_positions`.

### 3. Problema: Errores en la simulación con posiciones

**Síntoma**: La gestión de posiciones largas/cortas o el cálculo de stop-loss falla durante la simulación.

**Solución**: Verificar el flujo completo de apertura/cierre de posiciones y la lógica de stop-loss/trailing-stop.

## Procedimiento de Prueba

1. **Preparación**:
   - Crear conjuntos de datos OHLCV de prueba representativos
   - Implementar estrategias de prueba simples pero realistas

2. **Pruebas Unitarias Básicas**:
   - Verificar inicialización del motor
   - Probar cálculo de indicadores técnicos
   - Verificar generación de señales

3. **Pruebas de Integración**:
   - Evaluar el flujo completo desde datos → indicadores → señales → simulación
   - Verificar cálculo correcto de P&L, drawdown y otras métricas

4. **Casos de Prueba Específicos**:
   - Probar la gestión de posiciones (largas y cortas)
   - Verificar la activación de stop-loss y trailing stop
   - Probar optimización de parámetros
   - Verificar funcionamiento con múltiples activos y timeframes

## Comando de Ejecución

Para ejecutar todas las pruebas de backtesting:

```bash
python -m pytest tests/unit/test_backtesting.py -v
```

Para ejecutar pruebas específicas:

```bash
python -m pytest tests/unit/test_backtesting.py::test_backtest_run_simple -v
```

## Recomendaciones para Futuras Mejoras

1. Implementar pruebas con datos reales de mercado (no solo datos generados aleatoriamente)
2. Agregar casos de prueba para condiciones extremas de mercado
3. Verificar el rendimiento con conjuntos de datos grandes (miles de puntos)
4. Comparar resultados con otras implementaciones de backtesting para validación cruzada