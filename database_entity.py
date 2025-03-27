"""
Implementación de entidad especializada para base de datos en Sistema Genesis.

Este módulo implementa la entidad Kronos especializada en la administración y optimización
de la base de datos del sistema de trading.
"""

import os
import logging
import random
import time
import threading
import json
from typing import Dict, Any, List, Optional, Tuple
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor

from enhanced_simple_cosmic_trader import EnhancedCosmicTrader
from enhanced_cosmic_entity_mixin import EnhancedCosmicEntityMixin

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseEntity(EnhancedCosmicTrader, EnhancedCosmicEntityMixin):
    """
    Entidad especializada en la gestión y optimización de bases de datos.
    Extiende las capacidades de la entidad de trading para enfocarse en 
    el monitoreo, mantenimiento y optimización de la base de datos.
    """
    
    def __init__(self, name: str, role: str = "Database", father: str = "otoniel", 
                 frequency_seconds: int = 40, database_type: str = "postgres"):
        """
        Inicializar entidad de base de datos.
        
        Args:
            name: Nombre de la entidad
            role: Rol (siempre será "Database")
            father: Nombre del creador/dueño
            frequency_seconds: Período de ciclo de vida en segundos
            database_type: Tipo de base de datos (postgres o sqlite)
        """
        super().__init__(name, role, father, frequency_seconds)
        
        self.database_type = database_type
        self.connection = None
        self.cursor = None
        self.connected = False
        self.queries_executed = 0
        self.last_optimization = time.time()
        self.optimization_frequency = 3600  # Cada hora
        self.stats = {
            "queries_executed": 0,
            "optimizations_performed": 0,
            "errors_detected": 0,
            "indexes_created": 0,
            "vacuum_operations": 0
        }
        
        # Personalidad y rasgos específicos
        self.personality_traits = ["Metódico", "Minucioso", "Preciso"]
        self.emotional_volatility = 0.3  # Baja volatilidad emocional
        
        # Especializaciones
        self.specializations = {
            "Database Administration": 0.9,
            "Query Optimization": 0.8,
            "Index Management": 0.7,
            "Data Integrity": 0.9,
            "Performance Monitoring": 0.8
        }
        
        # Intentar establecer conexión con la base de datos
        self._setup_database_connection()
        
        logger.info(f"[{self.name}] Entidad de base de datos inicializada para {database_type}")
    
    def _setup_database_connection(self):
        """Establecer conexión con la base de datos."""
        try:
            if self.database_type == "postgres":
                # Usar variables de entorno para la conexión
                connection_params = {
                    "host": os.environ.get("PGHOST"),
                    "port": os.environ.get("PGPORT"),
                    "user": os.environ.get("PGUSER"),
                    "password": os.environ.get("PGPASSWORD"),
                    "database": os.environ.get("PGDATABASE")
                }
                
                # Verificar si tenemos los parámetros necesarios
                missing_params = [k for k, v in connection_params.items() if not v]
                if missing_params:
                    missing = ", ".join(missing_params)
                    logger.warning(f"[{self.name}] Faltan parámetros de conexión: {missing}")
                    return False
                
                # Conectar a PostgreSQL
                self.connection = psycopg2.connect(
                    host=connection_params["host"],
                    port=connection_params["port"],
                    user=connection_params["user"],
                    password=connection_params["password"],
                    database=connection_params["database"]
                )
                self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
                
            elif self.database_type == "sqlite":
                # Conectar a SQLite
                db_path = "cosmic_trading.db"
                self.connection = sqlite3.connect(db_path)
                self.connection.row_factory = sqlite3.Row
                self.cursor = self.connection.cursor()
            
            # Marcar como conectado si llegamos hasta aquí
            self.connected = True
            logger.info(f"[{self.name}] Conexión establecida con {self.database_type}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] Error al conectar a {self.database_type}: {str(e)}")
            self.connected = False
            return False
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[List[Dict[str, Any]]]:
        """
        Ejecutar consulta SQL y retornar resultados.
        
        Args:
            query: Consulta SQL
            params: Parámetros de la consulta
            
        Returns:
            Lista de resultados como diccionarios o None si hay error
        """
        if not self.connected:
            if not self._setup_database_connection():
                return None
        
        try:
            # Ejecutar la consulta
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            # Si es SELECT, retornar resultados
            if query.strip().upper().startswith("SELECT"):
                if self.database_type == "postgres":
                    results = self.cursor.fetchall()
                    return [dict(row) for row in results]
                else:  # sqlite
                    columns = [col[0] for col in self.cursor.description]
                    return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            # Si es DML (INSERT, UPDATE, DELETE), hacer commit
            elif any(query.strip().upper().startswith(op) for op in ["INSERT", "UPDATE", "DELETE"]):
                self.connection.commit()
                return {"affected_rows": self.cursor.rowcount}
            
            # Otro tipo de consulta
            else:
                self.connection.commit()
                return {"success": True}
                
        except Exception as e:
            logger.error(f"[{self.name}] Error ejecutando consulta: {str(e)}")
            self.stats["errors_detected"] += 1
            return None
        finally:
            self.queries_executed += 1
            self.stats["queries_executed"] += 1
    
    def optimize_database(self) -> Dict[str, Any]:
        """
        Realizar optimización periódica de la base de datos.
        
        Returns:
            Resultados de la optimización
        """
        results = {
            "optimized_tables": 0,
            "created_indexes": 0,
            "vacuum_operations": 0,
            "errors": 0,
            "success": False
        }
        
        if not self.connected:
            if not self._setup_database_connection():
                return results
        
        try:
            # Obtener tablas a optimizar
            if self.database_type == "postgres":
                tables_query = """
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public'
                """
            else:  # sqlite
                tables_query = """
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """
            
            tables = self.execute_query(tables_query)
            if not tables:
                return results
            
            table_names = [list(table.values())[0] for table in tables]
            
            # Optimizar cada tabla
            for table in table_names:
                # Analizar tabla
                if self.database_type == "postgres":
                    self.execute_query(f"ANALYZE {table}")
                    # Verificar si necesita vacuum
                    vacuum_needed = self.execute_query(f"""
                        SELECT pg_size_pretty(pg_total_relation_size('{table}')) AS size,
                               n_dead_tup AS dead_tuples
                        FROM pg_stat_user_tables
                        WHERE relname = '{table}'
                    """)
                    
                    if vacuum_needed and vacuum_needed[0].get("dead_tuples", 0) > 1000:
                        self.execute_query(f"VACUUM {table}")
                        results["vacuum_operations"] += 1
                        self.stats["vacuum_operations"] += 1
                
                elif self.database_type == "sqlite":
                    # SQLite no tiene ANALYZE como PostgreSQL
                    self.execute_query("PRAGMA optimize")
                    # Vacuum automático en SQLite
                    self.execute_query("VACUUM")
                    results["vacuum_operations"] += 1
                    self.stats["vacuum_operations"] += 1
                
                results["optimized_tables"] += 1
            
            # Marcar tiempo de última optimización
            self.last_optimization = time.time()
            results["success"] = True
            self.stats["optimizations_performed"] += 1
            
            # Mensaje de optimización
            optimization_message = self.generate_message(
                "optimización", 
                f"He optimizado {results['optimized_tables']} tablas y realizado {results['vacuum_operations']} operaciones de vacuum."
            )
            self.broadcast_message(optimization_message)
            
            return results
            
        except Exception as e:
            logger.error(f"[{self.name}] Error en optimización: {str(e)}")
            results["errors"] += 1
            self.stats["errors_detected"] += 1
            return results
    
    def monitor_performance(self) -> Dict[str, Any]:
        """
        Monitorear rendimiento de la base de datos.
        
        Returns:
            Métricas de rendimiento
        """
        metrics = {
            "connection_status": self.connected,
            "queries_executed": self.queries_executed,
            "last_optimization": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_optimization)),
            "database_type": self.database_type,
            "database_size": "Unknown",
            "active_connections": 0,
            "slow_queries": 0
        }
        
        if not self.connected:
            if not self._setup_database_connection():
                return metrics
        
        try:
            # Obtener tamaño de la base de datos
            if self.database_type == "postgres":
                size_query = """
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                """
                size_result = self.execute_query(size_query)
                if size_result:
                    metrics["database_size"] = size_result[0]["db_size"]
                
                # Obtener conexiones activas
                conn_query = """
                    SELECT count(*) as connections FROM pg_stat_activity 
                    WHERE state = 'active'
                """
                conn_result = self.execute_query(conn_query)
                if conn_result:
                    metrics["active_connections"] = conn_result[0]["connections"]
                
                # Obtener consultas lentas
                slow_query = """
                    SELECT count(*) as slow_count FROM pg_stat_activity 
                    WHERE state = 'active' AND now() - query_start > interval '5 seconds'
                """
                slow_result = self.execute_query(slow_query)
                if slow_result:
                    metrics["slow_queries"] = slow_result[0]["slow_count"]
            
            elif self.database_type == "sqlite":
                # SQLite no proporciona estas métricas de la misma manera
                # Podemos obtener el tamaño del archivo
                if os.path.exists("cosmic_trading.db"):
                    size_bytes = os.path.getsize("cosmic_trading.db")
                    metrics["database_size"] = f"{size_bytes / (1024*1024):.2f} MB"
            
            return metrics
            
        except Exception as e:
            logger.error(f"[{self.name}] Error monitoreando rendimiento: {str(e)}")
            return metrics
    
    def trade(self):
        """
        Implementar método trade requerido por la clase base abstracta.
        Para la entidad de base de datos, esto representa optimizar consultas
        y gestionar eficientemente la base de datos.
        """
        # Simular análisis de consultas y optimización
        trade_result = {
            "action": "optimize",
            "target": "database",
            "metrics": {
                "queries_optimized": self.stats.get("queries_executed", 0) * 0.1,
                "performance_gain": random.uniform(0.5, 5.0),
                "space_saved_kb": random.uniform(10, 100)
            }
        }
        
        # Registrar actividad de trading
        self.last_trade_time = time.time()
        self.trades_count += 1
        
        return trade_result
        
    def process_cycle(self):
        """
        Procesar ciclo de vida de la entidad de base de datos.
        Sobreescribe el método de la clase base.
        """
        if not self.is_alive:
            return
        
        # Actualizar ciclo base
        super().process_base_cycle()
        
        # Ciclo específico de entidad de base de datos
        try:
            # Estado inicial
            if not self.connected:
                self._setup_database_connection()
                return
            
            # Actualizar estado
            self.update_state()
            
            # Verificar si es momento de optimizar
            time_since_last_optimization = time.time() - self.last_optimization
            if time_since_last_optimization > self.optimization_frequency:
                self.optimize_database()
            
            # Monitorear rendimiento
            performance_metrics = self.monitor_performance()
            
            # Generar mensaje informativo sobre estado de la base de datos
            if random.random() < 0.2:  # 20% de probabilidad
                info_message = self.generate_database_insight()
                self.broadcast_message(info_message)
                
        except Exception as e:
            logger.error(f"[{self.name}] Error en ciclo de proceso: {str(e)}")
            self.handle_error(str(e))
    
    def generate_database_insight(self) -> str:
        """
        Generar insight sobre el estado de la base de datos.
        
        Returns:
            Mensaje con insight
        """
        insights = [
            f"La base de datos ha procesado {self.stats['queries_executed']} consultas desde mi activación.",
            f"He detectado {self.stats['errors_detected']} errores potenciales que he solucionado.",
            f"He realizado {self.stats['optimizations_performed']} optimizaciones para mantener el rendimiento óptimo.",
            f"Estado actual de la conexión: {'Activa' if self.connected else 'Inactiva'}.",
            f"Mi esencia {self.dominant_trait} me permite mantener la integridad de los datos con precisión."
        ]
        
        # Elegir un insight aleatorio
        insight = random.choice(insights)
        
        # Formatear como mensaje
        return self.generate_message("insight", insight)
    
    def handle_error(self, error_message: str):
        """
        Manejar error de base de datos.
        
        Args:
            error_message: Mensaje de error
        """
        # Registrar error
        logger.error(f"[{self.name}] Error detectado: {error_message}")
        self.stats["errors_detected"] += 1
        
        # Intentar reconectar si es un error de conexión
        if not self.connected or "connection" in error_message.lower():
            self._setup_database_connection()
        
        # Informar del error
        error_notification = self.generate_message(
            "error", 
            f"He detectado un error y estoy aplicando medidas correctivas: {error_message[:50]}..."
        )
        self.broadcast_message(error_notification)
    
    def update_state(self):
        """Actualizar estado interno basado en métricas de base de datos."""
        # Simulación de variación de estado basado en actividad
        energy_variation = 0
        
        # Perder energía por consultas ejecutadas
        energy_loss = self.queries_executed * 0.001
        energy_variation -= energy_loss
        
        # Ganar energía por optimizaciones
        if time.time() - self.last_optimization < 60:  # Optimización reciente
            energy_variation += 5
        
        # Ajustar nivel basado en estadísticas
        level_adjustment = (
            self.stats["optimizations_performed"] * 0.05 +
            self.stats["queries_executed"] * 0.0001 -
            self.stats["errors_detected"] * 0.1
        )
        
        # Aplicar cambios
        self.adjust_energy(energy_variation)
        self.adjust_level(level_adjustment)
        
        # Actualizar emoción basada en estado de la base de datos
        if not self.connected:
            self.emotion = "Preocupación"
        elif self.stats["errors_detected"] > 10:
            self.emotion = "Alerta"
        elif time.time() - self.last_optimization > self.optimization_frequency * 1.5:
            self.emotion = "Inquietud"
        else:
            emotions = ["Calma", "Satisfacción", "Precisión", "Equilibrio"]
            self.emotion = random.choice(emotions)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de la entidad para mostrar en UI.
        Extiende el método base con información específica de base de datos.
        
        Returns:
            Diccionario con estado
        """
        base_status = super().get_status()
        
        # Añadir métricas específicas de base de datos
        db_status = {
            "database_type": self.database_type,
            "connected": self.connected,
            "queries_executed": self.queries_executed,
            "last_optimization": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_optimization)),
            "stats": self.stats,
            "specializations": self.specializations
        }
        
        # Combinar estados
        combined_status = {**base_status, **db_status}
        return combined_status

def create_database_entity(name="Kronos", father="otoniel", frequency_seconds=40, database_type="postgres"):
    """
    Crear y configurar una entidad de base de datos.
    
    Args:
        name: Nombre de la entidad
        father: Nombre del creador/dueño
        frequency_seconds: Período de ciclo de vida en segundos
        database_type: Tipo de base de datos (postgres o sqlite)
        
    Returns:
        Instancia de DatabaseEntity
    """
    return DatabaseEntity(name, "Database", father, frequency_seconds, database_type)

if __name__ == "__main__":
    # Prueba básica de la entidad
    kronos = create_database_entity()
    print(f"Entidad {kronos.name} creada con rol {kronos.role}")
    
    # Iniciar ciclo de vida en un hilo separado
    thread = threading.Thread(target=kronos.start_lifecycle)
    thread.daemon = True
    thread.start()
    
    # Mantener vivo por un tiempo
    try:
        for i in range(10):
            time.sleep(1)
            print(f"Estado de {kronos.name}: Energía={kronos.energy:.1f}, Nivel={kronos.level:.1f}, Emoción={kronos.emotion}")
    
    except KeyboardInterrupt:
        print("Deteniendo prueba...")
    finally:
        # Detener ciclo de vida
        kronos.stop_lifecycle()
        print(f"Entidad {kronos.name} detenida")