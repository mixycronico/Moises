-- Creación de esquemas para el Sistema Genesis basados en archivos JSON

-- Tabla para resultados de pruebas de intensidad
CREATE TABLE IF NOT EXISTS gen_intensity_results (
    id SERIAL PRIMARY KEY,
    intensity FLOAT NOT NULL,
    average_success_rate FLOAT NOT NULL,
    average_essential_rate FLOAT NOT NULL,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para ciclos de procesamiento
CREATE TABLE IF NOT EXISTS gen_processing_cycles (
    id SERIAL PRIMARY KEY,
    cycle_id VARCHAR(100) NOT NULL,
    intensity FLOAT NOT NULL,
    success_rate FLOAT NOT NULL,
    essential_success_rate FLOAT NOT NULL,
    total_events INTEGER NOT NULL,
    successful_events INTEGER NOT NULL,
    essential_total INTEGER NOT NULL,
    essential_successful INTEGER NOT NULL,
    resultados_intensidad_id INTEGER REFERENCES gen_intensity_results(id) ON DELETE CASCADE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para componentes del sistema
CREATE TABLE IF NOT EXISTS gen_components (
    id SERIAL PRIMARY KEY,
    component_id VARCHAR(50) NOT NULL,
    processed INTEGER NOT NULL,
    success INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    radiation_emissions INTEGER NOT NULL,
    transmutations INTEGER NOT NULL,
    energy FLOAT NOT NULL,
    success_rate FLOAT NOT NULL,
    essential BOOLEAN NOT NULL,
    resultados_intensidad_id INTEGER REFERENCES gen_intensity_results(id) ON DELETE CASCADE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para estadísticas temporales
CREATE TABLE IF NOT EXISTS gen_temporal_stats (
    id SERIAL PRIMARY KEY,
    total_events INTEGER NOT NULL,
    past_events INTEGER NOT NULL,
    present_events INTEGER NOT NULL,
    future_events INTEGER NOT NULL,
    verify_continuity_total INTEGER NOT NULL,
    verify_continuity_success INTEGER NOT NULL,
    verify_continuity_failure INTEGER NOT NULL,
    induce_anomaly_total INTEGER NOT NULL,
    induce_anomaly_success INTEGER NOT NULL,
    induce_anomaly_failure INTEGER NOT NULL,
    record_total INTEGER NOT NULL,
    record_success INTEGER NOT NULL,
    record_failure INTEGER NOT NULL,
    protection_level INTEGER NOT NULL,
    initialized BOOLEAN NOT NULL,
    last_interaction FLOAT NOT NULL,
    time_since_last FLOAT NOT NULL,
    resultados_intensidad_id INTEGER REFERENCES gen_intensity_results(id) ON DELETE CASCADE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para detalles de eventos temporales
CREATE TABLE IF NOT EXISTS gen_temporal_events (
    id SERIAL PRIMARY KEY,
    timeline VARCHAR(20) NOT NULL, -- past, present, future
    event_type VARCHAR(100) NOT NULL,
    count INTEGER NOT NULL,
    estadisticas_temporales_id INTEGER REFERENCES gen_temporal_stats(id) ON DELETE CASCADE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para resultados de pruebas extremas
CREATE TABLE IF NOT EXISTS gen_extreme_test_results (
    id SERIAL PRIMARY KEY,
    start_time FLOAT NOT NULL,
    operations INTEGER NOT NULL,
    successes INTEGER NOT NULL,
    failures INTEGER NOT NULL,
    total_time FLOAT NOT NULL,
    success_rate FLOAT NOT NULL,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para resultados de singularidad
CREATE TABLE IF NOT EXISTS gen_singularity_results (
    id SERIAL PRIMARY KEY,
    nivel_singularidad FLOAT NOT NULL,
    tasa_exito FLOAT NOT NULL,
    operaciones_totales INTEGER NOT NULL, 
    tiempo_total FLOAT NOT NULL,
    transmutaciones_realizadas INTEGER NOT NULL,
    modo_trascendental VARCHAR(50) NOT NULL,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para estrategias adaptativas
CREATE TABLE IF NOT EXISTS gen_adaptive_strategies (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    capital_inicial FLOAT NOT NULL,
    umbral_eficiencia FLOAT NOT NULL,
    umbral_saturacion FLOAT NOT NULL,
    tipo_modelo VARCHAR(50) NOT NULL,
    activa BOOLEAN DEFAULT TRUE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para eficiencia de capital
CREATE TABLE IF NOT EXISTS gen_capital_efficiency (
    id SERIAL PRIMARY KEY,
    capital FLOAT NOT NULL,
    eficiencia FLOAT NOT NULL,
    simbolo VARCHAR(20) NOT NULL,
    estrategia_id INTEGER REFERENCES gen_adaptive_strategies(id) ON DELETE CASCADE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para puntos de saturación
CREATE TABLE IF NOT EXISTS gen_saturation_points (
    id SERIAL PRIMARY KEY,
    simbolo VARCHAR(20) NOT NULL,
    punto_saturacion FLOAT NOT NULL,
    confianza FLOAT NOT NULL,
    estrategia_id INTEGER REFERENCES gen_adaptive_strategies(id) ON DELETE CASCADE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para modelos de predicción
CREATE TABLE IF NOT EXISTS gen_prediction_models (
    id SERIAL PRIMARY KEY,
    simbolo VARCHAR(20) NOT NULL,
    tipo_modelo VARCHAR(50) NOT NULL,
    parametros JSONB NOT NULL,
    r_cuadrado FLOAT NOT NULL,
    estrategia_id INTEGER REFERENCES gen_adaptive_strategies(id) ON DELETE CASCADE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para registros de orquestración de estrategias
CREATE TABLE IF NOT EXISTS gen_strategy_orchestration (
    id SERIAL PRIMARY KEY,
    estrategia_seleccionada VARCHAR(100) NOT NULL,
    rendimiento FLOAT NOT NULL,
    simbolo VARCHAR(20) NOT NULL,
    senal VARCHAR(20) NOT NULL, -- buy, sell, hold
    timestamp FLOAT NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para clasificación de criptomonedas
CREATE TABLE IF NOT EXISTS gen_crypto_classification (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    score FLOAT NOT NULL,
    volatility_score FLOAT NOT NULL,
    volume_score FLOAT NOT NULL,
    trend_score FLOAT NOT NULL,
    momentum_score FLOAT NOT NULL,
    classification VARCHAR(20) NOT NULL, -- hot, neutral, cold
    timestamp FLOAT NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para métricas de criptomonedas
CREATE TABLE IF NOT EXISTS gen_crypto_metrics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    price FLOAT NOT NULL,
    volume_24h FLOAT NOT NULL,
    change_24h FLOAT NOT NULL,
    market_cap FLOAT NOT NULL,
    timestamp FLOAT NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para transacciones
CREATE TABLE IF NOT EXISTS gen_transactions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- buy, sell
    quantity FLOAT NOT NULL,
    price FLOAT NOT NULL,
    timestamp FLOAT NOT NULL,
    strategy_id VARCHAR(100) NOT NULL,
    fee FLOAT NOT NULL,
    status VARCHAR(20) NOT NULL, -- completed, failed, pending
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para operaciones de trading
CREATE TABLE IF NOT EXISTS gen_trading_operations (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    entry_price FLOAT NOT NULL,
    exit_price FLOAT NOT NULL,
    entry_time FLOAT NOT NULL,
    exit_time FLOAT NOT NULL,
    quantity FLOAT NOT NULL,
    profit_loss FLOAT NOT NULL,
    strategy_id VARCHAR(100) NOT NULL,
    success BOOLEAN NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para métricas de rendimiento
CREATE TABLE IF NOT EXISTS gen_performance_metrics (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(100) NOT NULL,
    win_rate FLOAT NOT NULL,
    profit_factor FLOAT NOT NULL,
    sharpe_ratio FLOAT NOT NULL,
    max_drawdown FLOAT NOT NULL,
    roi FLOAT NOT NULL,
    trades_count INTEGER NOT NULL,
    period_start FLOAT NOT NULL,
    period_end FLOAT NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para señales de trading
CREATE TABLE IF NOT EXISTS gen_trading_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20) NOT NULL, -- buy, sell, hold
    strength FLOAT NOT NULL,
    timestamp FLOAT NOT NULL,
    strategy_id VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    executed BOOLEAN DEFAULT FALSE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para asignación de capital
CREATE TABLE IF NOT EXISTS gen_capital_allocation (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    allocation_percentage FLOAT NOT NULL,
    capital_amount FLOAT NOT NULL,
    timestamp FLOAT NOT NULL,
    strategy_id VARCHAR(100) NOT NULL,
    optimizer_id VARCHAR(100) NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para logs del sistema
CREATE TABLE IF NOT EXISTS gen_system_logs (
    id SERIAL PRIMARY KEY,
    log_level VARCHAR(20) NOT NULL, -- info, warning, error, critical
    component VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    timestamp FLOAT NOT NULL,
    details JSONB,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para eventos del sistema
CREATE TABLE IF NOT EXISTS gen_system_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    source VARCHAR(100) NOT NULL,
    target VARCHAR(100) NOT NULL,
    timestamp FLOAT NOT NULL,
    data JSONB,
    processed BOOLEAN DEFAULT FALSE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para configuración del sistema
CREATE TABLE IF NOT EXISTS gen_system_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value JSONB NOT NULL,
    description TEXT,
    last_updated FLOAT NOT NULL,
    updated_by VARCHAR(100),
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para anomalías del sistema
CREATE TABLE IF NOT EXISTS gen_system_anomalies (
    id SERIAL PRIMARY KEY,
    anomaly_type VARCHAR(100) NOT NULL,
    severity INTEGER NOT NULL, -- 1-5
    component VARCHAR(100) NOT NULL,
    timestamp FLOAT NOT NULL,
    details JSONB,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_timestamp FLOAT,
    resolution_details TEXT,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para checkpoints del sistema
CREATE TABLE IF NOT EXISTS gen_system_checkpoints (
    id SERIAL PRIMARY KEY,
    checkpoint_id VARCHAR(100) NOT NULL UNIQUE,
    timestamp FLOAT NOT NULL,
    component VARCHAR(100) NOT NULL,
    data JSONB,
    checkpoint_type VARCHAR(50) NOT NULL, -- manual, auto, scheduled
    restored BOOLEAN DEFAULT FALSE,
    restored_timestamp FLOAT,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para latencias del sistema
CREATE TABLE IF NOT EXISTS gen_system_latencies (
    id SERIAL PRIMARY KEY,
    component VARCHAR(100) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    timestamp FLOAT NOT NULL,
    latency_ms FLOAT NOT NULL,
    success BOOLEAN NOT NULL,
    details JSONB,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para exchanges
CREATE TABLE IF NOT EXISTS gen_exchanges (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    api_key_encrypted VARCHAR(255),
    api_secret_encrypted VARCHAR(255),
    enabled BOOLEAN DEFAULT TRUE,
    testnet BOOLEAN DEFAULT TRUE,
    last_connected FLOAT,
    connection_status VARCHAR(50),
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para resiliencia del sistema
CREATE TABLE IF NOT EXISTS gen_system_resilience (
    id SERIAL PRIMARY KEY,
    component VARCHAR(100) NOT NULL,
    resilience_score FLOAT NOT NULL, -- 0-1
    circuit_state VARCHAR(50) NOT NULL, -- CLOSED, OPEN, HALF_OPEN, etc.
    failure_count INTEGER NOT NULL,
    success_count INTEGER NOT NULL,
    last_failure_timestamp FLOAT,
    timestamp FLOAT NOT NULL,
    details JSONB,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para monitoreo de recursos
CREATE TABLE IF NOT EXISTS gen_resource_monitoring (
    id SERIAL PRIMARY KEY,
    timestamp FLOAT NOT NULL,
    cpu_usage FLOAT NOT NULL, -- porcentaje
    memory_usage FLOAT NOT NULL, -- porcentaje
    disk_usage FLOAT NOT NULL, -- porcentaje
    network_in_bytes BIGINT NOT NULL,
    network_out_bytes BIGINT NOT NULL,
    active_connections INTEGER NOT NULL,
    active_threads INTEGER NOT NULL,
    details JSONB,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para indicadores técnicos
CREATE TABLE IF NOT EXISTS gen_technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value FLOAT NOT NULL,
    timestamp FLOAT NOT NULL,
    timeframe VARCHAR(20) NOT NULL, -- 1m, 5m, 15m, 1h, etc.
    parameters JSONB,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para riesgo de posiciones
CREATE TABLE IF NOT EXISTS gen_position_risk (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    position_size FLOAT NOT NULL,
    entry_price FLOAT NOT NULL,
    current_price FLOAT NOT NULL,
    stop_loss FLOAT,
    take_profit FLOAT,
    risk_ratio FLOAT NOT NULL, -- riesgo/recompensa
    risk_percentage FLOAT NOT NULL, -- porcentaje del capital
    timestamp FLOAT NOT NULL,
    strategy_id VARCHAR(100) NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para registro de llamadas 
CREATE TABLE IF NOT EXISTS gen_api_calls (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL, -- GET, POST, etc.
    timestamp FLOAT NOT NULL,
    response_time_ms FLOAT NOT NULL,
    status_code INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    ip_address VARCHAR(50),
    user_agent TEXT,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);