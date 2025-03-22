"""
Tests específicos para la configuración (core.settings).

Este módulo prueba las funcionalidades del sistema de configuración,
incluyendo gestión de valores, carga/guardado de archivos, valores encriptados,
y validación de esquemas.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch

from genesis.config.settings import Settings


@pytest.fixture
def settings():
    """Proporcionar una instancia de configuración para pruebas."""
    return Settings()


@pytest.fixture
def settings_file():
    """Proporcionar un archivo temporal para pruebas de configuración."""
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)  # Cerrar el descriptor de archivo que crea mkstemp
    
    yield path
    
    # Limpiar al finalizar
    if os.path.exists(path):
        os.unlink(path)


def test_settings_basics(settings):
    """Probar operaciones básicas de configuración."""
    # Nota: Settings ahora inicia con valores predeterminados
    # No podemos verificar que inicia vacía
    
    # Agregar valores y verificar
    settings.set("string_key", "value")
    settings.set("int_key", 123)
    settings.set("float_key", 3.14)
    settings.set("bool_key", True)
    settings.set("list_key", [1, 2, 3])
    settings.set("dict_key", {"a": 1, "b": 2})
    
    # Verificar valores individuales
    assert settings.get("string_key") == "value"
    assert settings.get("int_key") == 123
    assert settings.get("float_key") == 3.14
    assert settings.get("bool_key") is True
    assert settings.get("list_key") == [1, 2, 3]
    assert settings.get("dict_key") == {"a": 1, "b": 2}
    
    # Verificar obtención con valor por defecto
    assert settings.get("non_existent", "default") == "default"
    
    # Verificar obtención de todos los valores
    all_settings = settings.get_all()
    assert "string_key" in all_settings
    assert "int_key" in all_settings


def test_settings_nested_keys(settings):
    """Probar manejo de claves anidadas con notación de puntos."""
    # Establecer valores anidados
    settings.set("server.host", "localhost")
    settings.set("server.port", 8080)
    settings.set("database.credentials.username", "user")
    settings.set("database.credentials.password", "pass")
    
    # Verificar valores individuales
    assert settings.get("server.host") == "localhost"
    assert settings.get("server.port") == 8080
    assert settings.get("database.credentials.username") == "user"
    assert settings.get("database.credentials.password") == "pass"
    
    # Verificar obtención de sección completa
    server_settings = settings.get("server")
    assert isinstance(server_settings, dict)
    assert server_settings["host"] == "localhost"
    assert server_settings["port"] == 8080
    
    # Verificar sección más profunda
    credentials = settings.get("database.credentials")
    assert isinstance(credentials, dict)
    assert credentials["username"] == "user"
    assert credentials["password"] == "pass"


def test_settings_save_load(settings, settings_file):
    """Probar guardado y carga de configuración desde archivo."""
    # Establecer algunos valores
    settings.set("app_name", "Genesis")
    settings.set("version", "1.0.0")
    settings.set("debug", True)
    settings.set("logging.level", "INFO")
    
    # Guardar a archivo
    settings.save_to_file(settings_file)
    
    # Verificar que el archivo existe y tiene contenido
    assert os.path.exists(settings_file)
    with open(settings_file, 'r') as f:
        content = f.read()
        assert len(content) > 0
    
    # Crear nueva instancia y cargar
    settings2 = Settings()
    settings2.load_from_file(settings_file)
    
    # Verificar valores cargados
    assert settings2.get("app_name") == "Genesis"
    assert settings2.get("version") == "1.0.0"
    assert settings2.get("debug") is True
    assert settings2.get("logging.level") == "INFO"


def test_settings_load_nonexistent_file(settings):
    """Probar carga de archivo inexistente."""
    non_existent_file = "/tmp/non_existent_settings_file.json"
    
    # Asegurarse de que el archivo no existe
    if os.path.exists(non_existent_file):
        os.unlink(non_existent_file)
    
    # Intentar cargar
    with pytest.raises(FileNotFoundError):
        settings.load_from_file(non_existent_file)


def test_settings_load_invalid_json(settings, settings_file):
    """Probar carga de archivo con JSON inválido."""
    # Escribir JSON inválido
    with open(settings_file, 'w') as f:
        f.write("{invalid: json, content}")
    
    # Intentar cargar
    with pytest.raises(json.JSONDecodeError):
        settings.load_from_file(settings_file)


def test_settings_merge(settings):
    """Probar fusión de configuraciones."""
    # Configuración inicial
    settings.set("app_name", "Genesis")
    settings.set("version", "1.0.0")
    settings.set("database.host", "localhost")
    settings.set("database.port", 5432)
    
    # Nueva configuración para fusionar
    new_settings = {
        "version": "1.1.0",  # Actualizar existente
        "debug": True,       # Añadir nuevo
        "database": {
            "port": 5433,    # Actualizar anidado
            "username": "postgres"  # Añadir anidado
        }
    }
    
    # Fusionar
    settings.merge(new_settings)
    
    # Verificar resultados
    assert settings.get("app_name") == "Genesis"  # No cambió
    assert settings.get("version") == "1.1.0"     # Actualizado
    assert settings.get("debug") is True          # Nuevo
    assert settings.get("database.host") == "localhost"  # No cambió
    assert settings.get("database.port") == 5433         # Actualizado
    assert settings.get("database.username") == "postgres"  # Nuevo anidado


def test_settings_sensitive_values(settings, settings_file):
    """Probar manejo de valores sensibles."""
    # Establecer valores, algunos marcados como sensibles
    settings.set("api_key", "secret_api_key_123", sensitive=True)
    settings.set("username", "admin")
    settings.set("database.password", "db_password_456", sensitive=True)
    
    # Guardar a archivo
    settings.save_to_file(settings_file)
    
    # Leer el archivo directamente y verificar que los valores sensibles no están en texto plano
    with open(settings_file, 'r') as f:
        content = f.read()
        assert "secret_api_key_123" not in content
        assert "db_password_456" not in content
        assert "admin" in content  # Valor no sensible debe estar en texto plano
    
    # Cargar desde archivo
    settings2 = Settings()
    settings2.load_from_file(settings_file)
    
    # Verificar que los valores sensibles se pueden recuperar
    assert settings2.get("api_key") == "secret_api_key_123"
    assert settings2.get("database.password") == "db_password_456"


def test_settings_environment_override(settings):
    """Probar anulación de configuraciones con variables de entorno."""
    # Establecer valores iniciales
    settings.set("database.host", "localhost")
    settings.set("database.port", 5432)
    
    # Simular variables de entorno
    with patch.dict(os.environ, {
        "GENESIS_DATABASE_HOST": "dbserver.example.com",
        "GENESIS_DATABASE_PORT": "5433"
    }):
        # Configurar para usar variables de entorno
        settings.load_from_env(prefix="GENESIS_")
        
        # Verificar que los valores fueron anulados
        assert settings.get("database.host") == "dbserver.example.com"
        assert settings.get("database.port") == 5433  # Debería convertirse a entero


def test_settings_type_conversion(settings):
    """Probar conversión de tipos al cargar desde variables de entorno o strings."""
    # Simular variables de entorno con diferentes tipos
    with patch.dict(os.environ, {
        "GENESIS_INT_VALUE": "42",
        "GENESIS_FLOAT_VALUE": "3.14159",
        "GENESIS_BOOL_TRUE": "true",
        "GENESIS_BOOL_FALSE": "false",
        "GENESIS_LIST_VALUE": "[1, 2, 3]",
        "GENESIS_DICT_VALUE": '{"key": "value"}'
    }):
        # Cargar desde variables de entorno
        settings.load_from_env(prefix="GENESIS_")
        
        # Verificar conversión de tipos
        assert settings.get("int_value") == 42
        assert isinstance(settings.get("int_value"), int)
        
        assert settings.get("float_value") == 3.14159
        assert isinstance(settings.get("float_value"), float)
        
        assert settings.get("bool_true") is True
        assert isinstance(settings.get("bool_true"), bool)
        
        assert settings.get("bool_false") is False
        assert isinstance(settings.get("bool_false"), bool)
        
        assert settings.get("list_value") == [1, 2, 3]
        assert isinstance(settings.get("list_value"), list)
        
        assert settings.get("dict_value") == {"key": "value"}
        assert isinstance(settings.get("dict_value"), dict)


def test_settings_schema_validation(settings):
    """Probar validación de configuración contra esquema."""
    # Definir esquema
    schema = {
        "type": "object",
        "properties": {
            "server": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer", "minimum": 1024, "maximum": 65535}
                },
                "required": ["host", "port"]
            },
            "database": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_connections": {"type": "integer", "minimum": 1}
                },
                "required": ["url"]
            },
            "logging": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]}
                }
            }
        },
        "required": ["server", "database"]
    }
    
    # Configuración válida
    settings.set("server.host", "localhost")
    settings.set("server.port", 8080)
    settings.set("database.url", "postgres://localhost/mydb")
    settings.set("database.max_connections", 10)
    settings.set("logging.level", "INFO")
    
    # Validar - debería pasar
    assert settings.validate_schema(schema) is True
    
    # Cambiar a configuración inválida
    settings.set("server.port", 80)  # Puerto por debajo del mínimo
    
    # Validar - debería fallar
    with pytest.raises(ValueError):
        settings.validate_schema(schema)
    
    # Otra configuración inválida - falta campo requerido
    settings = Settings()
    settings.set("server.host", "localhost")
    settings.set("server.port", 8080)
    # Falta database.url
    
    # Validar - debería fallar
    with pytest.raises(ValueError):
        settings.validate_schema(schema)


def test_settings_default_values(settings):
    """Probar inicialización con valores por defecto."""
    # Definir valores por defecto
    defaults = {
        "app_name": "Genesis",
        "version": "1.0.0",
        "server": {
            "host": "0.0.0.0",
            "port": 8080
        },
        "database": {
            "url": "sqlite:///app.db",
            "pool_size": 5
        }
    }
    
    # Crear configuración con valores por defecto
    settings_with_defaults = Settings(defaults=defaults)
    
    # Verificar valores
    assert settings_with_defaults.get("app_name") == "Genesis"
    assert settings_with_defaults.get("server.host") == "0.0.0.0"
    assert settings_with_defaults.get("database.pool_size") == 5
    
    # Verificar que se pueden anular los valores por defecto
    settings_with_defaults.set("app_name", "Custom App")
    settings_with_defaults.set("server.port", 9000)
    
    assert settings_with_defaults.get("app_name") == "Custom App"
    assert settings_with_defaults.get("server.port") == 9000
    assert settings_with_defaults.get("server.host") == "0.0.0.0"  # No cambia


def test_settings_remove(settings):
    """Probar eliminación de valores de configuración."""
    # Establecer algunos valores
    settings.set("app_name", "Genesis")
    settings.set("version", "1.0.0")
    settings.set("server.host", "localhost")
    settings.set("server.port", 8080)
    
    # Eliminar uno
    settings.remove("version")
    
    # Verificar que se eliminó
    assert settings.get("version") is None
    
    # Verificar que los demás siguen
    assert settings.get("app_name") == "Genesis"
    
    # Eliminar valor anidado
    settings.remove("server.port")
    
    # Verificar eliminación anidada
    assert settings.get("server.port") is None
    assert settings.get("server.host") == "localhost"  # Este sigue


def test_settings_clear(settings):
    """Probar limpieza completa de configuración."""
    # Establecer algunos valores
    settings.set("app_name", "Genesis")
    settings.set("version", "1.0.0")
    
    # Verificar que hay valores
    assert len(settings.get_all()) > 0
    
    # Limpiar
    settings.clear()
    
    # Verificar que está vacío
    assert len(settings.get_all()) == 0


def test_settings_copy(settings):
    """Probar copia de configuración."""
    # Establecer algunos valores
    settings.set("app_name", "Genesis")
    settings.set("nested.value", 42)
    
    # Copiar
    copy = settings.copy()
    
    # Verificar que tiene los mismos valores
    assert copy.get("app_name") == "Genesis"
    assert copy.get("nested.value") == 42
    
    # Modificar el original
    settings.set("app_name", "Modified")
    
    # Verificar que la copia no se modificó
    assert copy.get("app_name") == "Genesis"


def test_settings_deep_copy(settings):
    """Probar copia profunda de objetos en la configuración."""
    # Establecer valor con objeto
    complex_obj = {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}}
    settings.set("complex", complex_obj)
    
    # Copiar
    copy = settings.copy()
    
    # Modificar objeto en el original
    settings.get("complex")["list"].append(4)
    settings.get("complex")["dict"]["c"] = 3
    
    # Verificar que la copia no se modificó
    assert 4 not in copy.get("complex")["list"]
    assert "c" not in copy.get("complex")["dict"]


def test_settings_iterator(settings):
    """Probar iteración sobre configuraciones."""
    # Establecer algunos valores
    settings.set("a", 1)
    settings.set("b", 2)
    settings.set("c", 3)
    
    # Iterar y recolectar claves
    keys = []
    for key in settings:
        keys.append(key)
    
    # Verificar que obtuvimos todas las claves
    assert sorted(keys) == ["a", "b", "c"]


def test_settings_contains(settings):
    """Probar operador 'in' para verificar existencia de claves."""
    # Establecer valores
    settings.set("existing_key", "value")
    settings.set("nested.key", "nested_value")
    
    # Verificar existencia
    assert "existing_key" in settings
    assert "nested.key" in settings
    assert "non_existing" not in settings


def test_settings_len(settings):
    """Probar función len() en configuraciones."""
    # Configuración vacía
    assert len(settings) == 0
    
    # Agregar valores
    settings.set("a", 1)
    settings.set("b", 2)
    settings.set("c.d", 3)  # Clave anidada
    
    # Verificar conteo
    assert len(settings) == 3


def test_settings_namespaces(settings):
    """Probar manejo de espacios de nombres en configuraciones."""
    # Establecer valores en diferentes espacios de nombres
    settings.set("app.name", "Genesis")
    settings.set("app.version", "1.0.0")
    settings.set("db.host", "localhost")
    settings.set("db.port", 5432)
    
    # Obtener todos los valores en un espacio de nombres
    app_settings = settings.get_namespace("app")
    
    # Verificar obtención correcta
    assert len(app_settings) == 2
    assert app_settings.get("name") == "Genesis"
    assert app_settings.get("version") == "1.0.0"
    
    # Obtener espacio de nombres diferente
    db_settings = settings.get_namespace("db")
    assert len(db_settings) == 2
    assert db_settings.get("host") == "localhost"
    assert db_settings.get("port") == 5432


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])