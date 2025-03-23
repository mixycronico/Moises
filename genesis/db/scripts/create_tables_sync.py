"""
Script para crear tablas en PostgreSQL usando psycopg2 de forma síncrona.

Este script lee el archivo SQL con las definiciones de tablas y las crea
en la base de datos PostgreSQL configurada en el sistema.
"""
import os
import logging
import psycopg2
from psycopg2 import sql

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_tables(sql_file_path):
    """
    Crear tablas en la base de datos a partir de un archivo SQL.
    
    Args:
        sql_file_path: Ruta al archivo SQL con las definiciones de tablas
        
    Returns:
        True si se crearon correctamente, False en caso de error
    """
    try:
        # Leer el archivo SQL
        logger.info(f"Leyendo archivo SQL: {sql_file_path}")
        with open(sql_file_path, 'r') as f:
            sql_content = f.read()
        
        # Dividir en comandos individuales
        sql_commands = []
        current_command = ""
        
        for line in sql_content.split('\n'):
            line = line.strip()
            
            # Ignorar comentarios y líneas vacías para el procesamiento
            if line.startswith('--') or not line:
                current_command += f"{line}\n"
                continue
                
            current_command += f"{line}\n"
            
            # Cuando llegue a un punto y coma, es el final del comando
            if line.endswith(';'):
                if current_command.strip():
                    # Agregar solo si hay contenido real (no solo comentarios)
                    has_content = False
                    for cmd_line in current_command.split('\n'):
                        if cmd_line.strip() and not cmd_line.strip().startswith('--'):
                            has_content = True
                            break
                    
                    if has_content:
                        sql_commands.append(current_command)
                        
                current_command = ""
        
        logger.info(f"Se encontraron {len(sql_commands)} comandos SQL")
        
        # Conectar a la base de datos
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            logger.error("No se encontró la variable de entorno DATABASE_URL")
            return False
        
        # Crear conexión a PostgreSQL
        conn = psycopg2.connect(db_url)
        conn.autocommit = True  # Para DDL es mejor usar autocommit
        cursor = conn.cursor()
        
        # Ejecutar cada comando SQL
        for i, cmd in enumerate(sql_commands, 1):
            try:
                logger.info(f"Ejecutando comando {i}/{len(sql_commands)}")
                # Imprimir las primeras 100 caracteres para facilitar depuración
                cmd_preview = cmd.strip().replace('\n', ' ')[:100] + ('...' if len(cmd.strip()) > 100 else '')
                logger.debug(f"SQL: {cmd_preview}")
                
                cursor.execute(cmd)
                logger.info(f"Comando {i} ejecutado correctamente")
            except Exception as e:
                logger.error(f"Error en comando {i}: {e}")
                logger.error(f"Comando SQL: {cmd}")
        
        # Verificar las tablas creadas
        try:
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = cursor.fetchall()
            if tables:
                logger.info(f"Tablas existentes en la base de datos: {len(tables)}")
                for table in tables:
                    logger.info(f"  - {table[0]}")
                return True
            else:
                logger.warning("No se encontraron tablas en la base de datos")
                return False
        except Exception as e:
            logger.error(f"Error al verificar tablas: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        logger.error(f"Error al crear tablas: {e}")
        return False

def main():
    """Función principal."""
    logger.info("Iniciando creación de tablas con el driver psycopg2")
    success = create_tables('create_tables_gen.sql')
    if success:
        logger.info("Tablas creadas correctamente")
    else:
        logger.error("Error al crear tablas")

if __name__ == "__main__":
    main()