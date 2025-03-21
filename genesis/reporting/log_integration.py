"""
Integración del sistema de logs con el generador de reportes.

Este módulo proporciona las funciones necesarias para recopilar, procesar
y visualizar los datos de logs en los informes generados por el sistema.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

# Importar funciones del gestor de logs
from genesis.utils.log_manager import query_logs, get_log_stats

class LogReportIntegration:
    """
    Clase para integrar datos de logs en los informes del sistema.
    
    Esta clase proporciona métodos para obtener y formatear datos de logs
    para ser incluidos en los informes diarios, semanales y mensuales.
    """
    
    def __init__(self, plot_dir: str = "data/plots"):
        """
        Inicializar la integración de logs.
        
        Args:
            plot_dir: Directorio para guardar los gráficos generados
        """
        self.logger = logging.getLogger("log_report_integration")
        self.plot_dir = plot_dir
        
        # Crear el directorio de gráficos si no existe
        os.makedirs(self.plot_dir, exist_ok=True)
    
    async def collect_logs_daily(self, date: datetime) -> Dict[str, Any]:
        """
        Recopilar logs para un informe diario.
        
        Args:
            date: Fecha del informe
            
        Returns:
            Diccionario con datos de logs para el informe
        """
        try:
            # Convertir fecha a formato ISO para la consulta
            start_date_str = date.replace(hour=0, minute=0, second=0).isoformat()
            end_date_str = date.replace(hour=23, minute=59, second=59).isoformat()
            
            # Consultar logs para este período
            logs_data = query_logs(
                start_date=start_date_str,
                end_date=end_date_str,
                limit=100
            )
            
            # Obtener estadísticas de logs
            log_stats = get_log_stats(start_date_str, end_date_str)
            
            result = {
                "total": len(logs_data),
                "by_level": log_stats.get("by_level", {}),
                "by_component": log_stats.get("by_component", {}),
                "entries": logs_data
            }
            
            self.logger.info(f"Se recopilaron {len(logs_data)} registros de logs para el informe diario.")
            return result
            
        except Exception as e:
            self.logger.error(f"Error al recopilar logs para el informe diario: {e}")
            # Devolver estructura vacía en caso de error
            return {
                "total": 0,
                "by_level": {},
                "by_component": {},
                "entries": []
            }
    
    async def collect_logs_weekly(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Recopilar logs para un informe semanal.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Diccionario con datos de logs para el informe
        """
        try:
            # Convertir fechas a formato ISO para la consulta
            start_date_str = start_date.replace(hour=0, minute=0, second=0).isoformat()
            end_date_str = end_date.replace(hour=23, minute=59, second=59).isoformat()
            
            # Consultar logs para este período
            logs_data = query_logs(
                start_date=start_date_str,
                end_date=end_date_str,
                limit=200  # Más logs para el período semanal
            )
            
            # Obtener estadísticas de logs
            log_stats = get_log_stats(start_date_str, end_date_str)
            
            # Procesamiento adicional para informes semanales
            # Agrupar logs por día para mostrar tendencia
            logs_by_day = {}
            for log in logs_data:
                log_date = datetime.fromisoformat(log.get("timestamp", "")).strftime("%Y-%m-%d")
                if log_date not in logs_by_day:
                    logs_by_day[log_date] = 0
                logs_by_day[log_date] += 1
            
            # Filtrar logs importantes (ERROR, CRITICAL, WARNING)
            important_logs = [log for log in logs_data if log.get("level") in ["ERROR", "CRITICAL", "WARNING"]]
            
            result = {
                "total": len(logs_data),
                "by_level": log_stats.get("by_level", {}),
                "by_component": log_stats.get("by_component", {}),
                "by_day": logs_by_day,
                "important": important_logs[:50],  # Limitar a 50 logs importantes
                "entries": logs_data[:100]  # Limitar a 100 logs recientes
            }
            
            self.logger.info(f"Se recopilaron {len(logs_data)} registros de logs para el informe semanal.")
            return result
            
        except Exception as e:
            self.logger.error(f"Error al recopilar logs para el informe semanal: {e}")
            # Devolver estructura vacía en caso de error
            return {
                "total": 0,
                "by_level": {},
                "by_component": {},
                "entries": []
            }
    
    async def collect_logs_monthly(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Recopilar logs para un informe mensual.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Diccionario con datos de logs para el informe
        """
        try:
            # Convertir fechas a formato ISO para la consulta
            start_date_str = start_date.replace(hour=0, minute=0, second=0).isoformat()
            end_date_str = end_date.replace(hour=23, minute=59, second=59).isoformat()
            
            # Consultar logs para este período
            logs_data = query_logs(
                start_date=start_date_str,
                end_date=end_date_str,
                limit=300  # Más logs para el período mensual
            )
            
            # Obtener estadísticas de logs
            log_stats = get_log_stats(start_date_str, end_date_str)
            
            # Procesamiento adicional para informes mensuales
            logs_by_day = {}
            logs_by_component = {}
            logs_by_level = {}
            
            for log in logs_data:
                # Agrupar por día
                log_date = datetime.fromisoformat(log.get("timestamp", "")).strftime("%Y-%m-%d")
                if log_date not in logs_by_day:
                    logs_by_day[log_date] = 0
                logs_by_day[log_date] += 1
                
                # Agrupar por componente
                component = log.get("component", "system")
                if component not in logs_by_component:
                    logs_by_component[component] = 0
                logs_by_component[component] += 1
                
                # Agrupar por nivel
                level = log.get("level", "INFO")
                if level not in logs_by_level:
                    logs_by_level[level] = 0
                logs_by_level[level] += 1
            
            # Generar gráficos de tendencia
            trend_chart_path = await self._generate_log_trend_chart(logs_by_day, start_date)
            
            result = {
                "total": len(logs_data),
                "by_level": log_stats.get("by_level", {}),
                "by_component": log_stats.get("by_component", {}),
                "by_day": logs_by_day,
                "trend_chart": trend_chart_path,
                "critical_events": [log for log in logs_data if log.get("level") in ["ERROR", "CRITICAL"]][:50],
                "entries": logs_data[:100]  # Limitar a 100 logs recientes
            }
            
            self.logger.info(f"Se recopilaron {len(logs_data)} registros de logs para el informe mensual.")
            return result
            
        except Exception as e:
            self.logger.error(f"Error al recopilar logs para el informe mensual: {e}")
            # Devolver estructura vacía en caso de error
            return {
                "total": 0,
                "by_level": {},
                "by_component": {},
                "entries": []
            }
    
    async def _generate_log_trend_chart(self, logs_by_day: Dict[str, int], 
                                       start_date: datetime) -> str:
        """
        Generar gráfico de tendencia de logs.
        
        Args:
            logs_by_day: Diccionario con conteo de logs por día
            start_date: Fecha de inicio del período
            
        Returns:
            Ruta del archivo de gráfico generado
        """
        try:
            # Ordenar fechas
            sorted_dates = sorted(logs_by_day.keys())
            counts = [logs_by_day[date] for date in sorted_dates]
            
            plt.figure(figsize=(12, 6))
            plt.bar(sorted_dates, counts, color='purple')
            plt.title(f"Tendencia de logs por día - {start_date.strftime('%b %Y')}")
            plt.xlabel("Fecha")
            plt.ylabel("Número de logs")
            plt.grid(axis='y', alpha=0.75)
            plt.xticks(rotation=45)
            
            # Guardar el gráfico
            chart_path = f"{self.plot_dir}/log_trend_{start_date.strftime('%Y%m')}.png"
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()
            
            return chart_path
        
        except Exception as e:
            self.logger.error(f"Error al generar gráfico de tendencia de logs: {e}")
            return ""