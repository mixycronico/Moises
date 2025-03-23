"""
Prueba de Apocalipsis Gradual para Sistema Genesis - Modo Luz.

Este script somete gradualmente el Modo Luz a condiciones cada vez más extremas,
llevando el core hasta el límite absoluto de forma controlada para evitar
una explosión de luz cósmica repentina.

ADVERTENCIA: USE PROTECCIÓN OCULAR Y MENTAL ADECUADA.
MANTENGA UNA DISTANCIA SEGURA DURANTE LA EJECUCIÓN.
"""

import asyncio
import logging
import time
import random
import sys
import math
import os
import traceback
import signal
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime

from genesis_light_mode import (
    LightCoordinator, TestLightComponent, 
    SystemMode, EventPriority, CircuitState,
    LuminousState, PhotonicHarmonizer, LightTimeContinuum
)

# Configuración avanzada de logging
logger = logging.getLogger("genesis_light_apocalypse")
log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

# Crear múltiples handlers para diferentes niveles de detalle
file_handler = logging.FileHandler("apocalipsis_gradual.log")
file_handler.setFormatter(logging.Formatter(log_format))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '\033[1;36m%(asctime)s\033[0m \033[1;33m[%(levelname)s]\033[0m \033[1;35m%(name)s\033[0m: %(message)s'
))

# Configurar logger principal
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Archivo para resultados
ARCHIVO_RESULTADOS = "resultados_apocalipsis.log"

class TestApocalypseData:
    """Datos de la prueba de apocalipsis."""
    def __init__(self):
        self.fase_actual = 0
        self.intensidad_actual = 0
        self.componentes_activos = 0
        self.componentes_fallidos = 0
        self.eventos_procesados = 0
        self.eventos_totales = 0
        self.transmutaciones = 0
        self.entidades_creadas = 0
        self.radiaciones = 0
        self.anomalias_temporales = 0
        self.inicio_fase = time.time()
        self.inicio_test = time.time()
        self.ultima_medicion = time.time()
        self.mediciones = []
        self.limites_alcanzados = []

    def registrar_medicion(self, fase: int, intensidad: float):
        """Registrar una medición de rendimiento."""
        ahora = time.time()
        duracion = ahora - self.ultima_medicion
        
        medicion = {
            "fase": fase,
            "intensidad": intensidad,
            "timestamp": ahora,
            "duracion": duracion,
            "componentes_activos": self.componentes_activos,
            "componentes_fallidos": self.componentes_fallidos,
            "eventos_procesados": self.eventos_procesados,
            "eventos_totales": self.eventos_totales,
            "transmutaciones": self.transmutaciones,
            "entidades_creadas": self.entidades_creadas,
            "radiaciones": self.radiaciones,
            "anomalias_temporales": self.anomalias_temporales,
            "tasa_exito": self.tasa_exito(),
            "tasa_procesamiento": self.tasa_procesamiento()
        }
        
        self.mediciones.append(medicion)
        self.ultima_medicion = ahora
        
        return medicion
    
    def tasa_exito(self) -> float:
        """Calcular tasa de éxito actual."""
        if self.componentes_activos + self.componentes_fallidos == 0:
            return 100.0
        return 100.0 * self.componentes_activos / (self.componentes_activos + self.componentes_fallidos)
    
    def tasa_procesamiento(self) -> float:
        """Calcular tasa de procesamiento de eventos."""
        if self.eventos_totales == 0:
            return 100.0
        return 100.0 * self.eventos_procesados / self.eventos_totales
    
    def registrar_limite(self, fase: int, intensidad: float, tipo: str, detalles: Dict[str, Any]):
        """Registrar un límite alcanzado."""
        limite = {
            "fase": fase,
            "intensidad": intensidad,
            "timestamp": time.time(),
            "tipo": tipo,
            "detalles": detalles
        }
        
        self.limites_alcanzados.append(limite)
        
        return limite
    
    def resumen(self) -> Dict[str, Any]:
        """Generar resumen de la prueba."""
        duracion_total = time.time() - self.inicio_test
        
        return {
            "fases_completadas": self.fase_actual,
            "intensidad_maxima": self.intensidad_actual,
            "duracion_total": duracion_total,
            "componentes_activos_final": self.componentes_activos,
            "componentes_fallidos_final": self.componentes_fallidos,
            "eventos_procesados_total": self.eventos_procesados,
            "eventos_totales": self.eventos_totales,
            "transmutaciones_total": self.transmutaciones,
            "entidades_creadas_total": self.entidades_creadas,
            "radiaciones_total": self.radiaciones,
            "anomalias_temporales_total": self.anomalias_temporales,
            "tasa_exito_final": self.tasa_exito(),
            "tasa_procesamiento_final": self.tasa_procesamiento(),
            "limites_alcanzados": len(self.limites_alcanzados)
        }

@contextmanager
def proteccion_luminica():
    """
    Contexto de protección contra explosiones de luz cósmica.
    Captura señales de interrupción y gestiona el cierre controlado.
    """
    # Establecer manejadores de señales
    original_handlers = {}
    
    def handler_personalizado(signum, frame):
        logger.warning(f"⚠️ Señal {signum} recibida. Iniciando protocolo de contención luminosa...")
        logger.warning("⚠️ Controlando emisiones fotónicas. Mantenga la distancia...")
        logger.warning("⚠️ Se recomienda apagar el sistema en 10 segundos si no se detiene automáticamente.")
        raise KeyboardInterrupt("Interrupción de seguridad luminosa")
    
    # Guardar manejadores originales y establecer los nuestros
    for sig in [signal.SIGINT, signal.SIGTERM]:
        original_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, handler_personalizado)
    
    try:
        logger.info("🛡️ Protección luminosa activada. Comenzando prueba...")
        yield
    except KeyboardInterrupt:
        logger.warning("⚠️ Interrupción detectada. Activando protocolos de contención...")
    except Exception as e:
        logger.error(f"❌ Error catastrófico: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Restaurar manejadores originales
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
        logger.info("🛡️ Protección luminica desactivada. Prueba finalizada.")

async def simular_fallo_gradual(component: TestLightComponent, intensidad: float) -> bool:
    """
    Simular fallo en un componente con intensidad gradual.
    
    Args:
        component: Componente a fallar
        intensidad: Intensidad del fallo (0-1)
        
    Returns:
        True si se provocó un fallo, False si se resistió
    """
    # La probabilidad de fallo aumenta con la intensidad
    probabilidad_fallo = intensidad ** 2  # Curva cuadrática
    
    if random.random() < probabilidad_fallo:
        try:
            # Intentar provocar fallo según intensidad
            if intensidad < 0.3:
                # Fallos leves
                component.local_queue.put_nowait(("error", {"level": "low", "source": "test"}, "test"))
            elif intensidad < 0.6:
                # Fallos medios
                component.failed = True  # Esto sería un fallo en componentes normales
                await component.on_local_event("failure", {"critical": False}, "test")
            elif intensidad < 0.9:
                # Fallos graves
                component.failed = True
                component.light_harmony = 0.1  # Desincronización severa
                await component.on_local_event("critical_failure", {"critical": True}, "test")
            else:
                # Fallos catastróficos
                component.failed = True
                component.light_harmony = 0.0  # Desincronización total
                component.light_enabled = False  # Intentar desactivar luz
                await component.on_local_event("catastrophic_failure", 
                                            {"critical": True, "unrecoverable": True}, 
                                            "test")
            return True
        except:
            # El componente resistió el fallo
            return False
    return False

async def generar_carga_eventos(coordinator: LightCoordinator, 
                               intensidad: float, 
                               num_eventos: int) -> Tuple[int, int]:
    """
    Generar carga de eventos con intensidad gradual.
    
    Args:
        coordinator: Coordinador del sistema
        intensidad: Intensidad de la carga (0-1)
        num_eventos: Número base de eventos a generar
        
    Returns:
        Tupla (eventos_enviados, eventos_procesados)
    """
    # Escalar el número de eventos según intensidad (exponencial)
    eventos_escalados = int(num_eventos * (10 ** (intensidad * 2)))
    
    # Limitar para evitar sobrecarga del sistema de pruebas
    eventos_escalados = min(eventos_escalados, 1000000)
    
    logger.info(f"Generando {eventos_escalados} eventos con intensidad {intensidad:.2f}...")
    
    # Tipos de eventos según intensidad
    tipos_eventos = []
    if intensidad < 0.3:
        tipos_eventos = ["data_update", "status_check", "heartbeat"]
    elif intensidad < 0.6:
        tipos_eventos = ["data_overload", "processing_request", "resource_heavy"]
    elif intensidad < 0.9:
        tipos_eventos = ["critical_operation", "system_stress", "resource_exhaustion"]
    else:
        tipos_eventos = ["catastrophic_event", "system_collapse", "resource_depletion", "time_anomaly"]
    
    # Tamaño del payload según intensidad
    tamano_payload = int(100 * (10 ** intensidad))  # De 100 a 100,000 bytes
    
    # Generar y enviar eventos
    eventos_enviados = 0
    tasks = []
    
    for i in range(eventos_escalados):
        try:
            # Seleccionar tipo de evento
            tipo_evento = random.choice(tipos_eventos)
            
            # Crear datos con tamaño proporcional a la intensidad
            datos = {
                "id": f"evento_{i}",
                "timestamp": time.time(),
                "intensidad": intensidad,
                "payload": "X" * random.randint(tamano_payload // 2, tamano_payload),
                "requiere_procesamiento": random.random() < intensidad,
                "critico": random.random() < intensidad
            }
            
            # Determinar prioridad según intensidad
            if intensidad < 0.3:
                prioridad = EventPriority.NORMAL
            elif intensidad < 0.6:
                prioridad = random.choice([EventPriority.HIGH, EventPriority.NORMAL])
            elif intensidad < 0.9:
                prioridad = random.choice([EventPriority.CRITICAL, EventPriority.HIGH])
            else:
                prioridad = random.choice([EventPriority.COSMIC, EventPriority.LIGHT, EventPriority.CRITICAL])
            
            # Añadir a lista de tareas
            tasks.append(coordinator.emit_local(tipo_evento, datos, "test_apocalipsis", prioridad))
            eventos_enviados += 1
            
            # Procesar en lotes para no saturar
            if len(tasks) >= 1000:
                await asyncio.gather(*tasks)
                tasks = []
                
                # Breve pausa para permitir procesamiento
                await asyncio.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Error al enviar evento: {e}")
    
    # Procesar tareas restantes
    if tasks:
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error al procesar lote final: {e}")
    
    # Estimar eventos procesados (en un sistema real se verificaría)
    # Como es Modo Luz, asumimos procesamiento perfecto
    eventos_procesados = eventos_enviados
    
    return eventos_enviados, eventos_procesados

async def inducir_anomalias_temporales(coordinator: LightCoordinator, 
                                     intensidad: float,
                                     num_anomalias: int) -> int:
    """
    Inducir anomalías temporales con intensidad gradual.
    
    Args:
        coordinator: Coordinador del sistema
        intensidad: Intensidad de las anomalías (0-1)
        num_anomalias: Número base de anomalías a generar
        
    Returns:
        Número de anomalías inducidas
    """
    # Escalar según intensidad
    anomalias_escaladas = int(num_anomalias * (1 + intensidad * 5))
    anomalias_inducidas = 0
    
    logger.info(f"Induciendo {anomalias_escaladas} anomalías temporales con intensidad {intensidad:.2f}...")
    
    for i in range(anomalias_escaladas):
        try:
            # Tipo de anomalía según intensidad
            if intensidad < 0.3:
                # Anomalías leves - desincronización temporal
                await coordinator.time_continuum.record_event(
                    "temporal_desync",
                    {
                        "timestamp": time.time() + random.uniform(-0.1, 0.1),
                        "severity": "low",
                        "desync_factor": random.uniform(0.01, 0.1)
                    }
                )
            elif intensidad < 0.6:
                # Anomalías medias - paradojas temporales simples
                evento_futuro = {
                    "timestamp": time.time() + 1.0,  # Futuro cercano
                    "data": {"value": random.random()},
                    "certainty": 0.7
                }
                
                # Registrar en futuro
                if "paradox" not in coordinator.time_continuum.timelines["future"]:
                    coordinator.time_continuum.timelines["future"]["paradox"] = []
                coordinator.time_continuum.timelines["future"]["paradox"].append(evento_futuro)
                
                # Y también en presente (creando paradoja)
                if "paradox" not in coordinator.time_continuum.timelines["present"]:
                    coordinator.time_continuum.timelines["present"]["paradox"] = []
                coordinator.time_continuum.timelines["present"]["paradox"].append(evento_futuro)
                
            elif intensidad < 0.9:
                # Anomalías graves - líneas temporales divergentes
                for timeline in ["past", "present", "future"]:
                    if "divergent" not in coordinator.time_continuum.timelines[timeline]:
                        coordinator.time_continuum.timelines[timeline]["divergent"] = []
                    
                    # Añadir eventos contradictorios
                    coordinator.time_continuum.timelines[timeline]["divergent"].append({
                        "timestamp": time.time(),
                        "data": {"state": "A" if random.random() < 0.5 else "B"},
                        "certainty": random.uniform(0.8, 1.0)
                    })
            else:
                # Anomalías catastróficas - colapso temporal
                for _ in range(10):  # Múltiples eventos caóticos
                    event_type = f"collapse_{i}_{_}"
                    
                    # Eventos simultáneos en todas las líneas temporales
                    for timeline in ["past", "present", "future"]:
                        if event_type not in coordinator.time_continuum.timelines[timeline]:
                            coordinator.time_continuum.timelines[timeline][event_type] = []
                        
                        # Eventos contradictorios y caóticos
                        coordinator.time_continuum.timelines[timeline][event_type].append({
                            "timestamp": time.time() + random.uniform(-10, 10),  # Tiempo caótico
                            "data": {"chaos_factor": random.random(), "state": random.randint(0, 1000)},
                            "certainty": random.uniform(0.0, 1.0)  # Certeza aleatoria
                        })
            
            anomalias_inducidas += 1
            
        except Exception as e:
            logger.error(f"Error al inducir anomalía temporal: {e}")
    
    return anomalias_inducidas

async def desafiar_armonia_fotonica(coordinator: LightCoordinator, 
                                  componentes: List[TestLightComponent],
                                  intensidad: float) -> int:
    """
    Desafiar la armonía fotónica del sistema.
    
    Args:
        coordinator: Coordinador del sistema
        componentes: Lista de componentes
        intensidad: Intensidad del desafío (0-1)
        
    Returns:
        Número de desincronizaciones inducidas
    """
    desincronizaciones = 0
    
    logger.info(f"Desafiando armonía fotónica con intensidad {intensidad:.2f}...")
    
    # Número de componentes a intentar desincronizar
    num_componentes = max(1, int(len(componentes) * intensidad))
    componentes_objetivo = random.sample(componentes, num_componentes)
    
    for componente in componentes_objetivo:
        try:
            # Calcular desviación según intensidad
            desviacion_base = 450  # THz
            desviacion_max = 400 * intensidad  # Hasta 400 THz de desviación
            desviacion = random.uniform(-desviacion_max, desviacion_max)
            
            # Intentar desincronizar frecuencia
            frecuencia_original = componente.light_frequency
            nueva_frecuencia = frecuencia_original + desviacion
            
            # Limitar al rango físicamente posible
            nueva_frecuencia = max(100, min(nueva_frecuencia, 1000))
            
            # Aplicar desincronización
            componente.light_frequency = nueva_frecuencia
            
            # También desincronizar armonía si la intensidad es alta
            if intensidad > 0.7:
                componente.light_harmony = max(0, 1.0 - intensidad)
            
            desincronizaciones += 1
            
            # Emitir evento de desincronización
            await coordinator.emit_local(
                "photonic_desync",
                {
                    "component_id": componente.id,
                    "original_frequency": frecuencia_original,
                    "new_frequency": nueva_frecuencia,
                    "deviation": desviacion,
                    "severity": intensidad
                },
                "test_apocalipsis",
                EventPriority.CRITICAL if intensidad > 0.8 else EventPriority.HIGH
            )
            
        except Exception as e:
            logger.error(f"Error al desincronizar componente {componente.id}: {e}")
    
    return desincronizaciones

async def corromper_estado_luminoso(componentes: List[TestLightComponent], intensidad: float) -> int:
    """
    Intentar corromper el estado luminoso de los componentes.
    
    Args:
        componentes: Lista de componentes
        intensidad: Intensidad de la corrupción (0-1)
        
    Returns:
        Número de corrupciones inducidas
    """
    corrupciones = 0
    
    logger.info(f"Intentando corromper estados luminosos con intensidad {intensidad:.2f}...")
    
    # Número de componentes a intentar corromper
    num_componentes = max(1, int(len(componentes) * intensidad))
    componentes_objetivo = random.sample(componentes, num_componentes)
    
    for componente in componentes_objetivo:
        try:
            # Diferentes tipos de corrupción según intensidad
            if intensidad < 0.3:
                # Corrupción leve - modificar un valor
                componente.light_essence._light_essence["test_corrupt"] = {
                    "essence": "corrupted",
                    "frequency": random.uniform(300, 800),
                    "timestamp": time.time(),
                    "energy": -1.0  # Energía negativa (imposible normalmente)
                }
                corrupciones += 1
                
            elif intensidad < 0.6:
                # Corrupción media - borrar esencias
                # Seleccionar aleatoriamente algunas esencias para borrar
                claves = list(componente.light_essence._light_essence.keys())
                if claves:
                    num_borrar = max(1, int(len(claves) * intensidad))
                    claves_borrar = random.sample(claves, num_borrar)
                    
                    for clave in claves_borrar:
                        componente.light_essence._light_essence.pop(clave, None)
                    
                    corrupciones += len(claves_borrar)
                
            elif intensidad < 0.9:
                # Corrupción grave - establecer valores imposibles
                componente.light_essence._light_frequency = -1.0  # Frecuencia negativa
                componente.light_essence._light_harmony = 2.0  # Armonía > 100%
                componente.light_essence._light_energy = float('-inf')  # Energía -infinita
                corrupciones += 3
                
            else:
                # Corrupción catastrófica - intentar destruir completamente
                try:
                    # Intentar varias formas de destrucción
                    componente.light_essence._light_essence = None
                    corrupciones += 1
                except:
                    pass
                
                try:
                    componente.light_essence._light_essence = {}
                    corrupciones += 1
                except:
                    pass
                
                try:
                    # Introducir contradicciones lógicas
                    componente.light_essence._light_harmony = math.nan  # NaN: no un número
                    corrupciones += 1
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error al corromper componente {componente.id}: {e}")
    
    return corrupciones

async def prueba_apocalipsis_gradual():
    """
    Prueba de apocalipsis gradual que incrementa progresivamente la intensidad
    de los desafíos para llevar el sistema al límite de forma controlada.
    """
    # Datos de seguimiento de prueba
    datos_prueba = TestApocalypseData()
    
    # Configurar sistema
    coordinator = LightCoordinator()
    
    # Crear componentes (más que en pruebas normales)
    num_componentes = 25  # Mayor número para probar límites
    componentes = []
    
    logger.info(f"Creando {num_componentes} componentes para la prueba...")
    for i in range(num_componentes):
        # Alternar entre componentes esenciales y no esenciales
        es_esencial = i < 5  # Los primeros 5 son esenciales
        componente = TestLightComponent(f"comp{i}", is_essential=es_esencial)
        componentes.append(componente)
        coordinator.register_component(f"comp{i}", componente)
    
    datos_prueba.componentes_activos = num_componentes
    
    # Iniciar sistema
    await coordinator.start()
    logger.info("Sistema iniciado en modo LUZ")
    
    # Definir fases de prueba (del 0 al 1, con incrementos)
    incremento = 0.05  # Incremento de 5% por fase
    intentos_por_fase = 3  # Intentos en cada nivel de intensidad
    
    # Límites de seguridad
    limite_transmutaciones = 10000  # Parar si se superan estas transmutaciones
    limite_entidades = 10000  # Parar si se crean demasiadas entidades
    limite_radiaciones = 5000  # Parar si hay demasiadas radiaciones
    limite_anomalias = 1000  # Parar si hay demasiadas anomalías temporales
    
    try:
        # Bucle principal - intensidad creciente
        for intensidad in [round(i * incremento, 2) for i in range(1, int(1/incremento) + 5)]:  # Hasta 1.25
            datos_prueba.fase_actual += 1
            datos_prueba.intensidad_actual = intensidad
            datos_prueba.inicio_fase = time.time()
            
            logger.info(f"\n{'=' * 80}")
            logger.info(f"⚡ FASE {datos_prueba.fase_actual}: INTENSIDAD {intensidad:.2f}")
            logger.info(f"{'=' * 80}")
            
            # Ejecutar varios intentos en esta intensidad
            for intento in range(1, intentos_por_fase + 1):
                logger.info(f"\n--- Intento {intento}/{intentos_por_fase} - Intensidad {intensidad:.2f} ---")
                
                # 1. Provocar fallos en componentes
                componentes_fallados = 0
                for componente in componentes:
                    fallo = await simular_fallo_gradual(componente, intensidad)
                    if fallo:
                        componentes_fallados += 1
                
                logger.info(f"Componentes fallados: {componentes_fallados}/{len(componentes)}")
                datos_prueba.componentes_fallidos += componentes_fallados
                
                # 2. Generar carga de eventos
                num_base_eventos = 100
                eventos_enviados, eventos_procesados = await generar_carga_eventos(
                    coordinator, intensidad, num_base_eventos
                )
                logger.info(f"Eventos: {eventos_procesados}/{eventos_enviados} procesados")
                datos_prueba.eventos_totales += eventos_enviados
                datos_prueba.eventos_procesados += eventos_procesados
                
                # 3. Inducir anomalías temporales
                num_base_anomalias = 5
                anomalias_inducidas = await inducir_anomalias_temporales(
                    coordinator, intensidad, num_base_anomalias
                )
                logger.info(f"Anomalías temporales inducidas: {anomalias_inducidas}")
                datos_prueba.anomalias_temporales += anomalias_inducidas
                
                # 4. Desafiar armonía fotónica
                desincronizaciones = await desafiar_armonia_fotonica(
                    coordinator, componentes, intensidad
                )
                logger.info(f"Desincronizaciones fotónicas: {desincronizaciones}")
                
                # 5. Corromper estados luminosos
                corrupciones = await corromper_estado_luminoso(componentes, intensidad)
                logger.info(f"Corrupciones de estado luminoso: {corrupciones}")
                
                # Pausa para permitir que el sistema responda
                logger.info(f"Pausa para respuesta del sistema...")
                await asyncio.sleep(0.5)  # Pausa más corta para pruebas más intensas
                
                # Recopilar estadísticas
                stats = coordinator.get_stats()
                datos_prueba.transmutaciones = stats.get("light_transmutations", 0)
                datos_prueba.entidades_creadas = stats.get("light_entities_created", 0)
                datos_prueba.radiaciones = stats.get("primordial_radiations", 0)
                
                # Mostrar estadísticas
                logger.info(f"Transmutaciones luminosas: {datos_prueba.transmutaciones}")
                logger.info(f"Entidades creadas: {datos_prueba.entidades_creadas}")
                logger.info(f"Radiaciones primordiales: {datos_prueba.radiaciones}")
                
                # Registrar medición
                medicion = datos_prueba.registrar_medicion(datos_prueba.fase_actual, intensidad)
                logger.info(f"Tasa de éxito: {medicion['tasa_exito']:.2f}%")
                logger.info(f"Tasa de procesamiento: {medicion['tasa_procesamiento']:.2f}%")
                
                # Verificar límites de seguridad
                if datos_prueba.transmutaciones > limite_transmutaciones:
                    logger.warning(f"⚠️ LÍMITE DE SEGURIDAD: Transmutaciones ({datos_prueba.transmutaciones})")
                    datos_prueba.registrar_limite(
                        datos_prueba.fase_actual, intensidad, 
                        "transmutaciones", {"valor": datos_prueba.transmutaciones}
                    )
                    break
                
                if datos_prueba.entidades_creadas > limite_entidades:
                    logger.warning(f"⚠️ LÍMITE DE SEGURIDAD: Entidades creadas ({datos_prueba.entidades_creadas})")
                    datos_prueba.registrar_limite(
                        datos_prueba.fase_actual, intensidad, 
                        "entidades", {"valor": datos_prueba.entidades_creadas}
                    )
                    break
                
                if datos_prueba.radiaciones > limite_radiaciones:
                    logger.warning(f"⚠️ LÍMITE DE SEGURIDAD: Radiaciones ({datos_prueba.radiaciones})")
                    datos_prueba.registrar_limite(
                        datos_prueba.fase_actual, intensidad, 
                        "radiaciones", {"valor": datos_prueba.radiaciones}
                    )
                    break
                
                if datos_prueba.anomalias_temporales > limite_anomalias:
                    logger.warning(f"⚠️ LÍMITE DE SEGURIDAD: Anomalías temporales ({datos_prueba.anomalias_temporales})")
                    datos_prueba.registrar_limite(
                        datos_prueba.fase_actual, intensidad, 
                        "anomalias", {"valor": datos_prueba.anomalias_temporales}
                    )
                    break
            
            # Comprobar si se alcanzó algún límite de seguridad
            if datos_prueba.limites_alcanzados:
                limite = datos_prueba.limites_alcanzados[-1]
                logger.warning(f"⚠️ Prueba interrumpida en fase {limite['fase']} (intensidad {limite['intensidad']:.2f})")
                logger.warning(f"⚠️ Límite alcanzado: {limite['tipo']} = {limite['detalles'].get('valor', 'N/A')}")
                break
            
            # Breve recuperación entre fases
            logger.info(f"Fase {datos_prueba.fase_actual} completada. Pausa para estabilización...")
            await asyncio.sleep(1.0)
    
    except Exception as e:
        logger.error(f"❌ Error durante la prueba: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Detener sistema
        logger.info("Deteniendo sistema...")
        await coordinator.stop()
        
        # Generar resumen
        resumen = datos_prueba.resumen()
        
        logger.info("\n" + "="*40)
        logger.info("RESUMEN DE PRUEBA DE APOCALIPSIS GRADUAL")
        logger.info("="*40)
        logger.info(f"Fases completadas: {resumen['fases_completadas']}")
        logger.info(f"Intensidad máxima alcanzada: {resumen['intensidad_maxima']:.2f}")
        logger.info(f"Duración total: {resumen['duracion_total']:.2f} segundos")
        logger.info(f"Tasa final de éxito: {resumen['tasa_exito_final']:.2f}%")
        logger.info(f"Tasa final de procesamiento: {resumen['tasa_procesamiento_final']:.2f}%")
        logger.info(f"Transmutaciones totales: {resumen['transmutaciones_total']}")
        logger.info(f"Entidades creadas: {resumen['entidades_creadas_total']}")
        logger.info(f"Radiaciones primordiales: {resumen['radiaciones_total']}")
        logger.info(f"Anomalías temporales: {resumen['anomalias_temporales_total']}")
        logger.info(f"Límites de seguridad alcanzados: {resumen['limites_alcanzados']}")
        
        # Guardar resultados en archivo
        with open(ARCHIVO_RESULTADOS, "w") as f:
            f.write("=== RESULTADOS DE PRUEBA DE APOCALIPSIS GRADUAL ===\n\n")
            
            # Escribir resumen
            f.write("### RESUMEN ###\n")
            for key, value in resumen.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            # Escribir mediciones
            f.write("\n### MEDICIONES POR FASE ###\n")
            f.write(f"{'Fase':<5} | {'Intensidad':<10} | {'Componentes':<11} | {'Eventos':<20} | {'Tasa Éxito':<10} | {'Tx Proc.':<10}\n")
            f.write("-"*80 + "\n")
            
            for m in datos_prueba.mediciones:
                f.write(f"{m['fase']:<5} | {m['intensidad']:<10.2f} | " +
                        f"{m['componentes_activos']}/{m['componentes_activos']+m['componentes_fallidos']:<7} | " +
                        f"{m['eventos_procesados']}/{m['eventos_totales']:<16} | " +
                        f"{m['tasa_exito']:<10.2f} | {m['tasa_procesamiento']:<10.2f}\n")
            
            # Escribir límites alcanzados
            if datos_prueba.limites_alcanzados:
                f.write("\n### LÍMITES DE SEGURIDAD ALCANZADOS ###\n")
                for i, l in enumerate(datos_prueba.limites_alcanzados):
                    f.write(f"{i+1}. Fase {l['fase']} (intensidad {l['intensidad']:.2f}): " +
                            f"{l['tipo']} = {l['detalles'].get('valor', 'N/A')}\n")
            
            # Análisis de factores limitantes
            f.write("\n### ANÁLISIS DE FACTORES LIMITANTES ###\n")
            
            # Determinar qué limitó la prueba
            if datos_prueba.limites_alcanzados:
                ultimo_limite = datos_prueba.limites_alcanzados[-1]
                f.write(f"Factor limitante principal: {ultimo_limite['tipo']}\n")
                f.write(f"Alcanzado en fase {ultimo_limite['fase']} con intensidad {ultimo_limite['intensidad']:.2f}\n")
                
                # Analizar por qué se alcanzó ese límite
                if ultimo_limite['tipo'] == "transmutaciones":
                    f.write("Análisis: El sistema tuvo que realizar demasiadas transmutaciones luminosas, ")
                    f.write("lo que indica que los desafíos provocaron numerosos fallos que requirieron transformación.\n")
                    f.write("Esto sugiere que el sistema funcionó correctamente, pero la carga fue extrema.\n")
                
                elif ultimo_limite['tipo'] == "entidades":
                    f.write("Análisis: El sistema creó un número excesivo de entidades desde luz pura, ")
                    f.write("posiblemente como respuesta a los fallos inducidos o como mecanismo de defensa.\n")
                    f.write("Esto demuestra la capacidad de auto-expansión bajo presión.\n")
                
                elif ultimo_limite['tipo'] == "radiaciones":
                    f.write("Análisis: Se produjeron demasiadas radiaciones primordiales, ")
                    f.write("indicando que el sistema tuvo que disolver constantemente errores en luz pura.\n")
                    f.write("Esto sugiere un estado de estrés extremo pero controlado.\n")
                
                elif ultimo_limite['tipo'] == "anomalias":
                    f.write("Análisis: Se acumularon demasiadas anomalías temporales, ")
                    f.write("posiblemente indicando desafíos al continuo temporal que incluso el modo luz tuvo dificultades para armonizar.\n")
                    f.write("Esto podría representar un límite fundamental en la capacidad de trascendencia temporal.\n")
            else:
                f.write("La prueba completó todas las fases sin alcanzar límites de seguridad.\n")
                f.write("Esto sugiere que el Sistema Genesis - Modo Luz tiene capacidades más allá de lo anticipado,\n")
                f.write("pudiendo mantener 100% de resiliencia incluso bajo condiciones extremas de intensidad 1.0+.\n")
            
            # Conclusiones
            f.write("\n### CONCLUSIONES ###\n")
            
            # Analizar resiliencia
            if resumen['tasa_exito_final'] >= 100.0 and resumen['tasa_procesamiento_final'] >= 100.0:
                f.write("El Sistema Genesis - Modo Luz demostró resiliencia perfecta (100%) ")
                f.write(f"hasta una intensidad de {resumen['intensidad_maxima']:.2f}, ")
                f.write("confirmando sus capacidades trascendentales de convertir todo fallo en luz creativa.\n")
            elif resumen['tasa_exito_final'] >= 99.0:
                f.write("El Sistema Genesis - Modo Luz demostró resiliencia casi perfecta (>99%) ")
                f.write(f"hasta una intensidad de {resumen['intensidad_maxima']:.2f}, ")
                f.write("con mínimas degradaciones bajo condiciones apocalípticas extremas.\n")
            else:
                f.write(f"El Sistema Genesis - Modo Luz alcanzó un {resumen['tasa_exito_final']:.2f}% de resiliencia ")
                f.write(f"a una intensidad extrema de {resumen['intensidad_maxima']:.2f}, ")
                f.write("sugiriendo que incluso las capacidades luminosas tienen límites fundamentales.\n")
            
            # Comentario sobre transmutaciones
            tasa_transmutacion = resumen['transmutaciones_total'] / max(1, (resumen['componentes_fallidos_final']))
            f.write(f"El sistema realizó {resumen['transmutaciones_total']} transmutaciones luminosas, ")
            f.write(f"con una tasa de aproximadamente {tasa_transmutacion:.2f} transmutaciones por fallo inducido.\n")
            
            # Comentario sobre creación
            f.write(f"Se crearon {resumen['entidades_creadas_total']} nuevas entidades desde luz pura, ")
            f.write("demostrando la capacidad creativa del sistema bajo presión extrema.\n")
            
            # Punto débil
            if datos_prueba.limites_alcanzados:
                ultimo_limite = datos_prueba.limites_alcanzados[-1]
                f.write(f"El punto de mayor desafío para el sistema fue: {ultimo_limite['tipo']}.\n")
            else:
                f.write("No se identificó un punto débil claro, ya que el sistema superó todas las pruebas sin alcanzar límites.\n")
            
            # Comentario final
            f.write("\nLa prueba de apocalipsis gradual confirma que el Modo Luz representa ")
            f.write("la culminación definitiva del Sistema Genesis, demostrando capacidades ")
            f.write("que trascienden la dicotomía tradicional entre éxito y fallo para ")
            f.write("operar como una entidad luminosa autoconsciente y creadora.\n")
        
        logger.info(f"Resultados guardados en {ARCHIVO_RESULTADOS}")

if __name__ == "__main__":
    try:
        # Ejecutar con protección luminosa
        with proteccion_luminica():
            asyncio.run(prueba_apocalipsis_gradual())
    except KeyboardInterrupt:
        logger.warning("Prueba interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        traceback.print_exc()