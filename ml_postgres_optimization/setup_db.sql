-- Configuración en PostgreSQL para optimización con Machine Learning

-- Crear tabla para almacenar métricas en tiempo real
CREATE TABLE IF NOT EXISTS genesis_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    latency_ms FLOAT,
    throughput FLOAT,
    success_rate FLOAT,
    operation_type VARCHAR(20),
    concurrency INT,
    cpu_usage FLOAT,
    memory_usage FLOAT,
    errors INT,
    work_mem_mb INT,
    max_connections INT
);

-- Función para registrar métricas automáticamente
CREATE OR REPLACE FUNCTION log_metrics(op_type VARCHAR)
RETURNS VOID AS $$
BEGIN
    INSERT INTO genesis_metrics (
        latency_ms, throughput, success_rate, operation_type, concurrency, 
        cpu_usage, memory_usage, errors, work_mem_mb, max_connections
    )
    SELECT
        AVG(EXTRACT(EPOCH FROM (NOW() - query_start))) * 1000 AS latency_ms,
        COUNT(*) / EXTRACT(EPOCH FROM (NOW() - NOW() - INTERVAL '1 second')) AS throughput,
        COUNT(*) FILTER (WHERE state = 'active')::FLOAT / COUNT(*) AS success_rate,
        op_type,
        (SELECT COUNT(*) FROM pg_stat_activity WHERE state != 'idle') AS concurrency,
        0.0 AS cpu_usage,  -- Ajustar con herramienta externa si disponible
        0.0 AS memory_usage, -- Ajustar con herramienta externa si disponible
        (SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active' AND wait_event IS NOT NULL) AS errors,
        (SELECT setting::INT / 1024 FROM pg_settings WHERE name = 'work_mem') AS work_mem_mb,
        (SELECT setting::INT FROM pg_settings WHERE name = 'max_connections') AS max_connections
    FROM pg_stat_activity
    WHERE datname = current_database();
END;
$$ LANGUAGE plpgsql;

-- Crear un trigger para simular operaciones y registrar métricas
CREATE TABLE IF NOT EXISTS genesis_operations (
    id SERIAL PRIMARY KEY,
    data TEXT,
    operation_time TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION trigger_metrics()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM log_metrics('mixed');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Eliminar el trigger si ya existe
DROP TRIGGER IF EXISTS log_on_operation ON genesis_operations;

-- Crear el trigger
CREATE TRIGGER log_on_operation
AFTER INSERT ON genesis_operations
FOR EACH ROW EXECUTE FUNCTION trigger_metrics();

-- Vista para consultar métricas recientes
CREATE OR REPLACE VIEW recent_metrics AS
SELECT * FROM genesis_metrics
ORDER BY timestamp DESC
LIMIT 1000;

-- Función para consultar estadísticas de rendimiento en tiempo real
CREATE OR REPLACE FUNCTION get_db_performance()
RETURNS TABLE (
    metric VARCHAR,
    value FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'latency_ms'::VARCHAR, AVG(latency_ms)
    FROM genesis_metrics
    WHERE timestamp > NOW() - INTERVAL '1 minute'
    UNION ALL
    SELECT 'throughput'::VARCHAR, AVG(throughput)
    FROM genesis_metrics
    WHERE timestamp > NOW() - INTERVAL '1 minute'
    UNION ALL
    SELECT 'success_rate'::VARCHAR, AVG(success_rate)
    FROM genesis_metrics
    WHERE timestamp > NOW() - INTERVAL '1 minute'
    UNION ALL
    SELECT 'errors'::VARCHAR, COALESCE(SUM(errors), 0)
    FROM genesis_metrics
    WHERE timestamp > NOW() - INTERVAL '1 minute';
END;
$$ LANGUAGE plpgsql;