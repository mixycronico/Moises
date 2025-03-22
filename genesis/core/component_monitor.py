"""
Monitoring de componentes del sistema Genesis.

Este módulo proporciona funcionalidades para monitorear la salud
de los componentes del sistema y aislar automáticamente aquellos
que presentan problemas para evitar fallos en cascada.
"""

import logging
import asyncio
from typing import Dict, Any, Optional

from tests.utils.timeout_helpers import emit_with_timeout, safe_get_response

# Configurar logger
logger = logging.getLogger(__name__)

class ComponentMonitor:
    """
    Monitor de salud de componentes.
    
    Esta clase se encarga de verificar periódicamente la salud de los
    componentes registrados en el motor de eventos y aislar aquellos
    que presentan problemas para evitar que afecten al resto del sistema.
    """
    
    def __init__(self, engine):
        """
        Inicializar el monitor de componentes.
        
        Args:
            engine: Instancia del motor de eventos
        """
        self.engine = engine
        self.health_status = {}
        self.isolation_status = {}
        self.monitor_task = None
        self.running = False
        
    async def start(self, check_interval: float = 5.0):
        """
        Iniciar el monitor de componentes.
        
        Args:
            check_interval: Intervalo en segundos entre verificaciones
        """
        if self.running:
            return
            
        self.running = True
        self.monitor_task = asyncio.create_task(
            self._monitor_loop(check_interval)
        )
        logger.info("Monitor de componentes iniciado")
        
    async def stop(self):
        """Detener el monitor de componentes."""
        if not self.running:
            return
            
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
            
        logger.info("Monitor de componentes detenido")
        
    async def _monitor_loop(self, check_interval: float):
        """
        Bucle principal del monitor.
        
        Args:
            check_interval: Intervalo en segundos entre verificaciones
        """
        try:
            while self.running:
                try:
                    await self.check_all_components()
                except Exception as e:
                    logger.error(f"Error en verificación de componentes: {e}")
                    
                # Esperar antes de la próxima verificación
                await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            logger.debug("Bucle de monitoreo cancelado")
        except Exception as e:
            logger.critical(f"Error crítico en bucle de monitoreo: {e}")
            
    async def check_all_components(self):
        """Verificar salud de todos los componentes."""
        if not hasattr(self.engine, 'components'):
            logger.warning("El motor no tiene componentes para verificar")
            return
            
        for component_id in list(self.engine.components.keys()):
            await self.check_component(component_id)
            
    async def check_component(self, component_id: str):
        """
        Verificar salud de un componente específico.
        
        Args:
            component_id: Identificador del componente a verificar
        """
        if self.isolation_status.get(component_id, False):
            # Componente ya aislado, verificar si debemos intentar rehabilitarlo
            await self._check_isolated_component(component_id)
            return
            
        # Verificar componente activo
        try:
            response = await emit_with_timeout(
                self.engine, 
                "check_status", 
                {}, 
                component_id,
                timeout=2.0,
                retries=1  # Un reintento por si hay problemas transitorios
            )
            
            healthy = safe_get_response(response, "healthy", False)
            error = safe_get_response(response, "error", None)
            
            self.health_status[component_id] = healthy
            
            if not healthy:
                logger.warning(f"Componente {component_id} no saludable: {error}")
                await self.isolate_component(component_id, error)
            else:
                logger.debug(f"Componente {component_id} saludable")
                
        except Exception as e:
            logger.error(f"Error al verificar componente {component_id}: {e}")
            # Si hay error en la verificación, considerar como no saludable
            self.health_status[component_id] = False
            await self.isolate_component(component_id, str(e))
            
    async def _check_isolated_component(self, component_id: str):
        """
        Verificar si un componente aislado puede ser rehabilitado.
        
        Args:
            component_id: Identificador del componente aislado
        """
        # Implementación básica: intentar verificar salud directamente
        try:
            response = await emit_with_timeout(
                self.engine, 
                "check_health", 
                {"direct_check": True}, 
                component_id,
                timeout=1.0
            )
            
            can_recover = safe_get_response(response, "can_recover", False)
            
            if can_recover:
                logger.info(f"Componente {component_id} puede ser rehabilitado")
                await self.rehabilitate_component(component_id)
                
        except Exception as e:
            logger.warning(f"Error al verificar recuperación de {component_id}: {e}")
            
    async def isolate_component(self, component_id: str, reason: Optional[str] = None):
        """
        Aislar un componente problemático.
        
        Args:
            component_id: Identificador del componente a aislar
            reason: Razón del aislamiento
        """
        if self.isolation_status.get(component_id, False):
            # Ya está aislado
            return
            
        logger.warning(f"Aislando componente {component_id}" + 
                      (f": {reason}" if reason else ""))
        
        self.isolation_status[component_id] = True
        
        # Notificar al sistema sobre el aislamiento
        await emit_with_timeout(
            self.engine,
            "component_isolated",
            {
                "component_id": component_id,
                "reason": reason or "unknown"
            },
            "component_monitor",
            timeout=1.0
        )
        
        # Opcionalmente, usar el motor directamente para desactivar el componente
        if hasattr(self.engine, 'pause_component'):
            try:
                await self.engine.pause_component(component_id)
            except Exception as e:
                logger.error(f"Error al pausar componente {component_id}: {e}")
                
    async def rehabilitate_component(self, component_id: str):
        """
        Rehabilitar un componente previamente aislado.
        
        Args:
            component_id: Identificador del componente a rehabilitar
        """
        if not self.isolation_status.get(component_id, False):
            # No estaba aislado
            return
            
        logger.info(f"Rehabilitando componente {component_id}")
        
        self.isolation_status[component_id] = False
        self.health_status[component_id] = True
        
        # Notificar al sistema sobre la rehabilitación
        await emit_with_timeout(
            self.engine,
            "component_rehabilitated",
            {"component_id": component_id},
            "component_monitor",
            timeout=1.0
        )
        
        # Opcionalmente, usar el motor directamente para reactivar el componente
        if hasattr(self.engine, 'resume_component'):
            try:
                await self.engine.resume_component(component_id)
            except Exception as e:
                logger.error(f"Error al reactivar componente {component_id}: {e}")
                
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Obtener el estado de salud general del sistema.
        
        Returns:
            Diccionario con información de salud del sistema
        """
        total_components = len(self.health_status)
        healthy_components = sum(1 for status in self.health_status.values() if status)
        isolated_components = sum(1 for status in self.isolation_status.values() if status)
        
        return {
            "total_components": total_components,
            "healthy_components": healthy_components,
            "isolated_components": isolated_components,
            "health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0,
            "component_status": {
                component_id: {
                    "healthy": self.health_status.get(component_id, False),
                    "isolated": self.isolation_status.get(component_id, False)
                }
                for component_id in self.health_status
            }
        }