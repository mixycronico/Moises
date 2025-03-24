-- Script para simular carga sostenida que genera datos para entrenamiento del modelo ML

DO $$
DECLARE
    i INTEGER;
    total INTEGER := 10000; -- Simular 10,000 operaciones para generar datos de entrenamiento
BEGIN
    RAISE NOTICE 'Iniciando simulación de carga con % operaciones...', total;
    
    FOR i IN 1..total LOOP
        -- Insertar operación para activar el trigger de métricas
        INSERT INTO genesis_operations (data)
        VALUES ('Test operation ' || i);
        
        -- Pequeña pausa para simular carga realista
        PERFORM pg_sleep(0.001);
        
        -- Mostrar progreso cada 1000 operaciones
        IF i % 1000 = 0 THEN
            RAISE NOTICE 'Progreso: % operaciones completadas', i;
        END IF;
    END LOOP;
    
    RAISE NOTICE 'Simulación de carga completada con éxito';
END;
$$;

-- Comando para exportar datos (ejecutar desde psql)
-- \COPY (SELECT * FROM genesis_metrics ORDER BY timestamp DESC LIMIT 5000) TO '/tmp/genesis_metrics.csv' WITH CSV HEADER;