"""
Pruebas de manejo de datos anómalos para el core del sistema Genesis.

Este módulo contiene pruebas que verifican la capacidad del sistema para
manejar datos corruptos, malformados o inesperados sin fallar, incluyendo
validación de datos, límites y tipos de datos incorrectos.
"""

import pytest
import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Union, Set
import json

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class DataValidatorComponent(Component):
    """Componente que valida datos de eventos y registra anomalías."""
    
    def __init__(self, name: str):
        """Inicializar componente validador."""
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.anomalies: List[Dict[str, Any]] = []
        self.total_processed = 0
        self.validation_errors = 0
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Iniciando componente validador {self.name}")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Deteniendo componente validador {self.name}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar y validar un evento, registrando anomalías.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
        """
        # Registrar el evento
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        })
        
        self.total_processed += 1
        
        # Validar datos según reglas
        issues = self.validate_data(event_type, data)
        
        # Si hay problemas, registrar anomalía
        if issues:
            self.anomalies.append({
                "type": event_type,
                "data": data,
                "source": source,
                "issues": issues,
                "timestamp": time.time()
            })
            self.validation_errors += 1
        
        return None
    
    def validate_data(self, event_type: str, data: Dict[str, Any]) -> List[str]:
        """
        Validar datos según el tipo de evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            
        Returns:
            Lista de problemas encontrados
        """
        issues = []
        
        # Verificar tipo de datos
        if not isinstance(data, dict):
            issues.append(f"Los datos no son un diccionario: {type(data)}")
            return issues
        
        # Validaciones básicas según tipo de evento
        if event_type.startswith("numeric_"):
            # Eventos numéricos deben tener valores numéricos
            for key, value in data.items():
                if key.startswith("num_") and not isinstance(value, (int, float)):
                    issues.append(f"El campo '{key}' debería ser numérico, pero es {type(value)}")
                if key == "positive" and isinstance(value, (int, float)) and value <= 0:
                    issues.append(f"El campo 'positive' debería ser positivo, pero es {value}")
        
        elif event_type.startswith("string_"):
            # Eventos de cadena deben tener valores de texto
            for key, value in data.items():
                if key.startswith("str_") and not isinstance(value, str):
                    issues.append(f"El campo '{key}' debería ser una cadena, pero es {type(value)}")
                if key == "non_empty" and isinstance(value, str) and value == "":
                    issues.append("El campo 'non_empty' no debería estar vacío")
        
        elif event_type.startswith("required_"):
            # Eventos con campos requeridos
            required_fields = data.get("required_fields", [])
            for field in required_fields:
                if field not in data:
                    issues.append(f"Campo requerido '{field}' no está presente")
        
        elif event_type.startswith("format_"):
            # Eventos con formatos específicos
            if "date" in data and isinstance(data["date"], str):
                # Validar formato ISO de fecha
                if not data["date"].startswith("20") or len(data["date"]) != 10:
                    issues.append(f"Formato de fecha inválido: {data['date']}")
            
            if "email" in data and isinstance(data["email"], str):
                # Validación simple de email
                if "@" not in data["email"] or "." not in data["email"]:
                    issues.append(f"Formato de email inválido: {data['email']}")
        
        elif event_type.startswith("limit_"):
            # Eventos con límites
            if "max_length" in data and isinstance(data.get("value"), str):
                max_length = data.get("max_length", 10)
                if len(data["value"]) > max_length:
                    issues.append(f"Valor excede la longitud máxima: {len(data['value'])} > {max_length}")
            
            if "range" in data and isinstance(data.get("value"), (int, float)):
                min_val = data.get("min", 0)
                max_val = data.get("max", 100)
                if data["value"] < min_val or data["value"] > max_val:
                    issues.append(f"Valor fuera de rango: {data['value']} no está entre {min_val} y {max_val}")
        
        return issues
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "total_processed": self.total_processed,
            "validation_errors": self.validation_errors,
            "error_rate": self.validation_errors / self.total_processed if self.total_processed > 0 else 0,
            "anomaly_count": len(self.anomalies),
            "unique_anomaly_types": len({a["type"] for a in self.anomalies})
        }


class TypeSensitiveComponent(Component):
    """Componente que espera tipos de datos específicos y falla si no los recibe."""
    
    def __init__(self, name: str):
        """Inicializar componente sensible a tipos."""
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.total_processed = 0
        self.error_count = 0
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Iniciando componente sensible a tipos {self.name}")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Deteniendo componente sensible a tipos {self.name}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, fallando si los tipos no son los esperados.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
            
        Raises:
            TypeError: Si los tipos de datos no son los esperados
            ValueError: Si los valores no son válidos
        """
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        })
        
        self.total_processed += 1
        
        try:
            # Validar según el tipo de evento (mucho más estricto que el validador)
            if event_type == "int_event":
                if "value" not in data:
                    raise KeyError("Campo 'value' requerido en evento int_event")
                if not isinstance(data["value"], int):
                    raise TypeError(f"Campo 'value' debe ser entero, pero es {type(data['value'])}")
                # Usar el valor para operaciones
                result = data["value"] * 2
                return {"result": result}
            
            elif event_type == "float_event":
                if "value" not in data:
                    raise KeyError("Campo 'value' requerido en evento float_event")
                if not isinstance(data["value"], float):
                    raise TypeError(f"Campo 'value' debe ser flotante, pero es {type(data['value'])}")
                # Usar el valor para operaciones
                result = data["value"] * 3.14
                return {"result": result}
            
            elif event_type == "string_event":
                if "value" not in data:
                    raise KeyError("Campo 'value' requerido en evento string_event")
                if not isinstance(data["value"], str):
                    raise TypeError(f"Campo 'value' debe ser cadena, pero es {type(data['value'])}")
                # Usar el valor para operaciones
                result = data["value"].upper()
                return {"result": result}
            
            elif event_type == "list_event":
                if "values" not in data:
                    raise KeyError("Campo 'values' requerido en evento list_event")
                if not isinstance(data["values"], list):
                    raise TypeError(f"Campo 'values' debe ser lista, pero es {type(data['values'])}")
                # Usar el valor para operaciones
                result = sum(data["values"]) if all(isinstance(x, (int, float)) for x in data["values"]) else 0
                return {"result": result}
            
            elif event_type == "complex_event":
                # Evento con estructura anidada
                if "nested" not in data:
                    raise KeyError("Campo 'nested' requerido en evento complex_event")
                if not isinstance(data["nested"], dict):
                    raise TypeError(f"Campo 'nested' debe ser diccionario, pero es {type(data['nested'])}")
                
                nested = data["nested"]
                if "id" not in nested or not isinstance(nested["id"], int):
                    raise ValueError("Campo 'nested.id' debe ser un entero")
                if "name" not in nested or not isinstance(nested["name"], str):
                    raise ValueError("Campo 'nested.name' debe ser una cadena")
                
                return {"result": f"ID: {nested['id']}, Name: {nested['name']}"}
            
        except (TypeError, ValueError, KeyError) as e:
            # Registrar error
            self.errors.append({
                "type": event_type,
                "data": data,
                "error": str(e),
                "timestamp": time.time()
            })
            self.error_count += 1
            # Re-lanzar excepción
            raise
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "total_processed": self.total_processed,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.total_processed if self.total_processed > 0 else 0
        }


class RobustComponent(Component):
    """Componente que maneja cualquier tipo de datos sin fallar."""
    
    def __init__(self, name: str):
        """Inicializar componente robusto."""
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.anomalies: List[Dict[str, Any]] = []
        self.total_processed = 0
        self.converted_events = 0
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Iniciando componente robusto {self.name}")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Deteniendo componente robusto {self.name}")
    
    async def handle_event(self, event_type: str, data: Any, source: str) -> Optional[Any]:
        """
        Manejar cualquier evento sin fallar, convirtiendo los datos si es necesario.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento (cualquier tipo)
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
        """
        # Convertir datos a formato compatible
        processed_data = self.ensure_dict(data)
        
        # Registrar el evento original y el procesado
        event_record = {
            "type": event_type,
            "original_data": data,
            "processed_data": processed_data,
            "source": source,
            "timestamp": time.time()
        }
        
        # Si los datos fueron convertidos, registrar anomalía
        if processed_data != data:
            self.anomalies.append(event_record)
            self.converted_events += 1
        
        self.events.append(event_record)
        self.total_processed += 1
        
        # Siempre procesar los datos sin fallar
        return {"status": "processed", "original_type": type(data).__name__}
    
    def ensure_dict(self, data: Any) -> Dict[str, Any]:
        """
        Convertir cualquier dato a un diccionario.
        
        Args:
            data: Datos a convertir
            
        Returns:
            Diccionario con los datos convertidos
        """
        if isinstance(data, dict):
            return data
        elif isinstance(data, (str, bytes)):
            # Intentar convertir de JSON
            try:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    return {"value": parsed}
            except:
                # Si no es JSON, devolver como valor
                return {"value": data}
        elif isinstance(data, (list, tuple)):
            # Convertir lista a diccionario
            return {"values": list(data)}
        elif isinstance(data, (int, float, bool)):
            # Valores simples
            return {"value": data}
        elif data is None:
            return {}
        else:
            # Cualquier otro tipo, intentar convertir sus atributos
            try:
                # Intentar usar los atributos del objeto como diccionario
                result = {key: value for key, value in data.__dict__.items() 
                         if not key.startswith('_')}
                if result:
                    return result
            except:
                pass
            
            # Si todo falla, al menos guardar el tipo
            return {"value": str(data), "type": type(data).__name__}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "total_processed": self.total_processed,
            "converted_events": self.converted_events,
            "conversion_rate": self.converted_events / self.total_processed if self.total_processed > 0 else 0,
            "unique_original_types": len({type(e["original_data"]).__name__ for e in self.events})
        }


@pytest.mark.asyncio
async def test_data_validation():
    """Prueba que el sistema puede validar y manejar datos anómalos."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componente validador
    validator = DataValidatorComponent("validator")
    
    # Registrar componente
    engine.register_component(validator)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos con datos válidos
    valid_events = [
        ("numeric_event", {"num_value": 10, "positive": 5}),
        ("string_event", {"str_value": "test", "non_empty": "value"}),
        ("required_event", {"required_fields": ["id", "name"], "id": 1, "name": "test"}),
        ("format_event", {"date": "2023-01-01", "email": "test@example.com"}),
        ("limit_event", {"value": "short", "max_length": 10})
    ]
    
    for event_type, data in valid_events:
        await engine.emit_event(event_type, data, "test")
        await asyncio.sleep(0.1)
    
    # Enviar eventos con datos inválidos
    invalid_events = [
        ("numeric_event", {"num_value": "not_a_number", "positive": -5}),
        ("string_event", {"str_value": 123, "non_empty": ""}),
        ("required_event", {"required_fields": ["id", "name"], "id": 1}),
        ("format_event", {"date": "01/01/2023", "email": "invalid-email"}),
        ("limit_event", {"value": "this is too long", "max_length": 5})
    ]
    
    for event_type, data in invalid_events:
        await engine.emit_event(event_type, data, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que todos los eventos se procesaron
    metrics = validator.get_metrics()
    assert metrics["total_processed"] == 10, "Deberían haberse procesado 10 eventos en total"
    
    # Verificar que se detectaron anomalías en los eventos inválidos
    assert metrics["validation_errors"] == 5, "Deberían haberse detectado 5 eventos con errores de validación"
    
    # Verificar que cada evento inválido generó al menos una anomalía
    assert metrics["anomaly_count"] >= 5, "Cada evento inválido debería generar al menos una anomalía"
    
    # Log de anomalías encontradas
    logger.info(f"Anomalías encontradas: {validator.anomalies}")
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_type_sensitive_component():
    """Prueba componentes que son sensibles a tipos de datos específicos."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    sensitive = TypeSensitiveComponent("sensitive")
    robust = RobustComponent("robust")
    
    # Registrar componentes
    engine.register_component(sensitive)
    engine.register_component(robust)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos válidos
    valid_events = [
        ("int_event", {"value": 42}),
        ("float_event", {"value": 3.14}),
        ("string_event", {"value": "hello"}),
        ("list_event", {"values": [1, 2, 3, 4, 5]}),
        ("complex_event", {"nested": {"id": 1, "name": "test"}})
    ]
    
    for event_type, data in valid_events:
        await engine.emit_event(event_type, data, "test")
        await asyncio.sleep(0.1)
    
    # Enviar eventos inválidos que causarán errores en el componente sensible
    invalid_events = [
        ("int_event", {"value": "42"}),  # String en lugar de int
        ("float_event", {"value": 3}),   # Int en lugar de float
        ("string_event", {"value": 123}), # Int en lugar de string
        ("list_event", {"values": "not a list"}), # String en lugar de list
        ("complex_event", {"nested": "not a dict"}) # String en lugar de dict
    ]
    
    for event_type, data in invalid_events:
        await engine.emit_event(event_type, data, "test")
        await asyncio.sleep(0.1)
    
    # Verificar métricas del componente sensible
    sensitive_metrics = sensitive.get_metrics()
    assert sensitive_metrics["total_processed"] == 10, "El componente sensible debería haber procesado 10 eventos"
    assert sensitive_metrics["error_count"] == 5, "El componente sensible debería haber registrado 5 errores"
    
    # Verificar métricas del componente robusto
    robust_metrics = robust.get_metrics()
    assert robust_metrics["total_processed"] == 10, "El componente robusto debería haber procesado 10 eventos"
    assert robust_metrics["converted_events"] == 0, "El componente robusto no debería haber convertido ningún evento (todos eran dicts)"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_non_dict_data():
    """Prueba el sistema con datos que no son diccionarios."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    validator = DataValidatorComponent("validator")
    robust = RobustComponent("robust")
    
    # Registrar componentes
    engine.register_component(validator)
    engine.register_component(robust)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos con tipos de datos que no son diccionarios
    non_dict_events = [
        ("string_data", "This is just a string"),
        ("int_data", 42),
        ("float_data", 3.14),
        ("list_data", [1, 2, 3, 4, 5]),
        ("none_data", None),
        ("bool_data", True)
    ]
    
    for event_type, data in non_dict_events:
        await engine.emit_event(event_type, data, "test")
        await asyncio.sleep(0.1)
    
    # Verificar métricas del validador
    validator_metrics = validator.get_metrics()
    assert validator_metrics["validation_errors"] == 6, "El validador debería haber encontrado errores en todos los eventos no-dict"
    
    # Verificar que el componente robusto procesó todos los eventos y los convirtió
    robust_metrics = robust.get_metrics()
    assert robust_metrics["total_processed"] == 6, "El componente robusto debería haber procesado 6 eventos"
    assert robust_metrics["converted_events"] == 6, "El componente robusto debería haber convertido 6 eventos"
    assert robust_metrics["unique_original_types"] == 6, "Debería haber 6 tipos de datos originales diferentes"
    
    # Log de conversiones
    logger.info("Conversiones realizadas por el componente robusto:")
    for anomaly in robust.anomalies:
        logger.info(f"Original ({type(anomaly['original_data']).__name__}): {anomaly['original_data']} -> Procesado: {anomaly['processed_data']}")
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_malformed_json():
    """Prueba el sistema con datos JSON malformados."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componente robusto
    robust = RobustComponent("robust")
    
    # Registrar componente
    engine.register_component(robust)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar cadenas JSON válidas
    valid_json_events = [
        ("valid_json1", '{"name": "test", "value": 42}'),
        ("valid_json2", '{"list": [1, 2, 3], "nested": {"key": "value"}}')
    ]
    
    for event_type, data in valid_json_events:
        await engine.emit_event(event_type, data, "test")
        await asyncio.sleep(0.1)
    
    # Enviar cadenas JSON inválidas
    invalid_json_events = [
        ("invalid_json1", '{"name": "test", value: 42}'),  # Falta comillas en key
        ("invalid_json2", '{"list": [1, 2, 3], "nested": {"key": "value}'),  # Falta comilla
        ("invalid_json3", 'not json at all')  # No es JSON
    ]
    
    for event_type, data in invalid_json_events:
        await engine.emit_event(event_type, data, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que todos los eventos se procesaron
    metrics = robust.get_metrics()
    assert metrics["total_processed"] == 5, "Deberían haberse procesado 5 eventos en total"
    assert metrics["converted_events"] == 5, "Todos los eventos deberían haberse convertido (eran strings)"
    
    # Verificar que los JSON válidos se convirtieron correctamente
    converted_valid = [e for e in robust.events if e["type"] in ("valid_json1", "valid_json2")]
    for event in converted_valid:
        assert isinstance(event["processed_data"], dict), f"JSON válido debería convertirse a dict: {event}"
        assert "value" in event["processed_data"] or "list" in event["processed_data"], f"JSON procesado debería conservar keys: {event}"
    
    # Verificar que los JSON inválidos se manejaron sin fallar
    converted_invalid = [e for e in robust.events if e["type"] in ("invalid_json1", "invalid_json2", "invalid_json3")]
    for event in converted_invalid:
        assert isinstance(event["processed_data"], dict), f"Incluso JSON inválido debería dar un dict: {event}"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_extreme_data_values():
    """Prueba el sistema con valores de datos extremos."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componente validador
    validator = DataValidatorComponent("validator")
    
    # Registrar componente
    engine.register_component(validator)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos con valores extremos
    extreme_events = [
        # Valores numéricos extremos
        ("numeric_event", {"num_value": 10**100, "positive": 10**100}),  # Número muy grande
        ("numeric_event", {"num_value": 10**-100, "positive": 10**-100}),  # Número muy pequeño
        ("numeric_event", {"num_value": 0, "positive": 0}),  # Cero (debería fallar en positive)
        
        # Strings extremos
        ("string_event", {"str_value": "a" * 1000000, "non_empty": "x"}),  # String muy largo
        ("string_event", {"str_value": "", "non_empty": "x"}),  # String vacío
        ("string_event", {"str_value": "\u0000\u0001\u0002", "non_empty": "x"}),  # Caracteres de control
        
        # Límites
        ("limit_event", {"value": "x" * 1000, "max_length": 10}),  # Muy por encima del límite
        ("limit_event", {"value": 1000000, "min": 0, "max": 100, "range": True}),  # Muy por encima del rango
        ("limit_event", {"value": -1000000, "min": 0, "max": 100, "range": True})  # Muy por debajo del rango
    ]
    
    for event_type, data in extreme_events:
        await engine.emit_event(event_type, data, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que todos los eventos se procesaron
    metrics = validator.get_metrics()
    assert metrics["total_processed"] == 9, "Deberían haberse procesado 9 eventos en total"
    
    # Al menos algunos de estos eventos deberían tener problemas de validación
    assert metrics["validation_errors"] > 0, "Deberían haberse detectado errores de validación en valores extremos"
    
    # Log de anomalías encontradas
    logger.info(f"Anomalías en valores extremos: {validator.anomalies}")
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_circular_references():
    """Prueba el sistema con datos que contienen referencias circulares."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componente robusto
    robust = RobustComponent("robust")
    
    # Registrar componente
    engine.register_component(robust)
    
    # Iniciar motor
    await engine.start()
    
    # Crear datos con referencias circulares
    circular_data1 = {"name": "circular1"}
    circular_data1["self_reference"] = circular_data1
    
    circular_data2 = {"name": "circular2", "child": {"name": "child"}}
    circular_data2["child"]["parent"] = circular_data2
    
    # Intentar enviar eventos con referencias circulares
    # Nota: Esto podría fallar en el motor, dependiendo de la implementación
    try:
        await engine.emit_event("circular_event1", circular_data1, "test")
        await asyncio.sleep(0.1)
    except Exception as e:
        logger.warning(f"Error al enviar evento con referencia circular: {e}")
    
    try:
        await engine.emit_event("circular_event2", circular_data2, "test")
        await asyncio.sleep(0.1)
    except Exception as e:
        logger.warning(f"Error al enviar evento con referencia circular anidada: {e}")
    
    # Enviar evento normal después para verificar que el sistema sigue funcionando
    await engine.emit_event("normal_event", {"name": "normal"}, "test")
    await asyncio.sleep(0.1)
    
    # Verificar que al menos el evento normal se procesó
    metrics = robust.get_metrics()
    assert metrics["total_processed"] >= 1, "Al menos el evento normal debería haberse procesado"
    
    # Log de eventos procesados
    logger.info(f"Eventos procesados: {robust.events}")
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_mixed_component_tolerance():
    """Prueba diferentes niveles de tolerancia en componentes que reciben los mismos eventos."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes con diferentes niveles de tolerancia
    strict = TypeSensitiveComponent("strict")
    validator = DataValidatorComponent("validator")
    robust = RobustComponent("robust")
    
    # Registrar componentes
    engine.register_component(strict)
    engine.register_component(validator)
    engine.register_component(robust)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos con diferentes problemas
    mixed_events = [
        ("int_event", {"value": 42}),  # Válido para todos
        ("int_event", {"value": "42"}),  # Inválido para strict, ok para otros
        ("int_event", [1, 2, 3]),  # Inválido para strict y validator, ok para robust
        ("limit_event", {"value": "too long", "max_length": 5}),  # Inválido por longitud
        ("numeric_event", {"num_value": "not_numeric"})  # Inválido por tipo
    ]
    
    for event_type, data in mixed_events:
        await engine.emit_event(event_type, data, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que el componente robusto procesó todos los eventos
    robust_metrics = robust.get_metrics()
    assert robust_metrics["total_processed"] == 5, "El componente robusto debería procesar todos los eventos"
    
    # Verificar que el validador procesó varios eventos pero encontró problemas
    validator_metrics = validator.get_metrics()
    assert validator_metrics["total_processed"] == 5, "El validador debería procesar todos los eventos"
    assert validator_metrics["validation_errors"] > 0, "El validador debería encontrar problemas en algunos eventos"
    
    # Verificar que el componente estricto tuvo errores
    strict_metrics = strict.get_metrics()
    assert strict_metrics["total_processed"] == 5, "El componente estricto debería intentar procesar todos los eventos"
    assert strict_metrics["error_count"] > 0, "El componente estricto debería tener errores en algunos eventos"
    
    # Detener motor
    await engine.stop()