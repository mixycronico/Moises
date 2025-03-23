-- Creación de esquemas para el Sistema Genesis basados en archivos JSON

-- Tabla para resultados de pruebas de intensidad
CREATE TABLE IF NOT EXISTS resultados_intensidad (
    id SERIAL PRIMARY KEY,
    intensity FLOAT NOT NULL,
    average_success_rate FLOAT NOT NULL,
    average_essential_rate FLOAT NOT NULL,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para ciclos de procesamiento
CREATE TABLE IF NOT EXISTS ciclos_procesamiento (
    id SERIAL PRIMARY KEY,
    cycle_id VARCHAR(100) NOT NULL,
    intensity FLOAT NOT NULL,
    success_rate FLOAT NOT NULL,
    essential_success_rate FLOAT NOT NULL,
    total_events INTEGER NOT NULL,
    successful_events INTEGER NOT NULL,
    essential_total INTEGER NOT NULL,
    essential_successful INTEGER NOT NULL,
    resultados_intensidad_id INTEGER REFERENCES resultados_intensidad(id) ON DELETE CASCADE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para componentes del sistema
CREATE TABLE IF NOT EXISTS componentes (
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
    resultados_intensidad_id INTEGER REFERENCES resultados_intensidad(id) ON DELETE CASCADE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para estadísticas temporales
CREATE TABLE IF NOT EXISTS estadisticas_temporales (
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
    resultados_intensidad_id INTEGER REFERENCES resultados_intensidad(id) ON DELETE CASCADE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para detalles de eventos temporales
CREATE TABLE IF NOT EXISTS eventos_temporales (
    id SERIAL PRIMARY KEY,
    timeline VARCHAR(20) NOT NULL, -- past, present, future
    event_type VARCHAR(100) NOT NULL,
    count INTEGER NOT NULL,
    estadisticas_temporales_id INTEGER REFERENCES estadisticas_temporales(id) ON DELETE CASCADE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para resultados de pruebas extremas
CREATE TABLE IF NOT EXISTS resultados_prueba_extrema (
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
CREATE TABLE IF NOT EXISTS resultados_singularidad (
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
CREATE TABLE IF NOT EXISTS estrategias_adaptativas (
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
CREATE TABLE IF NOT EXISTS eficiencia_capital (
    id SERIAL PRIMARY KEY,
    capital FLOAT NOT NULL,
    eficiencia FLOAT NOT NULL,
    simbolo VARCHAR(20) NOT NULL,
    estrategia_id INTEGER REFERENCES estrategias_adaptativas(id) ON DELETE CASCADE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para puntos de saturación
CREATE TABLE IF NOT EXISTS puntos_saturacion (
    id SERIAL PRIMARY KEY,
    simbolo VARCHAR(20) NOT NULL,
    punto_saturacion FLOAT NOT NULL,
    confianza FLOAT NOT NULL,
    estrategia_id INTEGER REFERENCES estrategias_adaptativas(id) ON DELETE CASCADE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para modelos de predicción
CREATE TABLE IF NOT EXISTS modelos_prediccion (
    id SERIAL PRIMARY KEY,
    simbolo VARCHAR(20) NOT NULL,
    tipo_modelo VARCHAR(50) NOT NULL,
    parametros JSONB NOT NULL,
    r_cuadrado FLOAT NOT NULL,
    estrategia_id INTEGER REFERENCES estrategias_adaptativas(id) ON DELETE CASCADE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para registros de orquestración de estrategias
CREATE TABLE IF NOT EXISTS orquestracion_estrategias (
    id SERIAL PRIMARY KEY,
    estrategia_seleccionada VARCHAR(100) NOT NULL,
    rendimiento FLOAT NOT NULL,
    simbolo VARCHAR(20) NOT NULL,
    senal VARCHAR(20) NOT NULL, -- buy, sell, hold
    timestamp FLOAT NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);