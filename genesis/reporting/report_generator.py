"""
Generador de reportes automatizados para el sistema Genesis.

Este módulo proporciona funcionalidades para generar informes periódicos
sobre el rendimiento del sistema, estrategias de trading, y métricas financieras.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import os
import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication

from genesis.core.base import Component
from genesis.utils.logger import setup_logging
from genesis.utils.log_manager import query_logs, get_log_stats
from genesis.notifications.email_notifier import email_notifier


class ReportGenerator(Component):
    """
    Generador de reportes automatizados.
    
    Este componente genera informes periódicos sobre el rendimiento
    del sistema, estrategias de trading, y métricas financieras.
    """
    
    def __init__(self, name: str = "report_generator"):
        """
        Inicializar el generador de reportes.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Configuración
        self.report_dir = "data/reports"
        self.plot_dir = f"{self.report_dir}/plots"
        
        # Crear directorios si no existen
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Destinatarios para informes
        self.report_recipients = []
        
        # Informes configurados
        self.configured_reports = {
            "daily": {
                "enabled": True,
                "time": "23:59",
                "recipients": []
            },
            "weekly": {
                "enabled": True,
                "day": 4,  # 0 = lunes, 6 = domingo (4 = viernes)
                "time": "20:00",
                "recipients": []
            },
            "monthly": {
                "enabled": True,
                "day": 1,  # Primer día del mes
                "time": "12:00",
                "recipients": []
            }
        }
    
    async def start(self) -> None:
        """Iniciar el generador de reportes."""
        await super().start()
        self.logger.info("Generador de reportes iniciado")
    
    async def stop(self) -> None:
        """Detener el generador de reportes."""
        await super().stop()
        self.logger.info("Generador de reportes detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Procesar eventos relevantes para reportes
        if event_type == "system.generate_report":
            report_type = data.get("report_type", "daily")
            recipients = data.get("recipients", self.report_recipients)
            
            if report_type == "daily":
                await self.generate_daily_report(recipients)
            elif report_type == "weekly":
                await self.generate_weekly_report(recipients)
            elif report_type == "monthly":
                await self.generate_monthly_report(recipients)
            elif report_type == "custom":
                start_date = data.get("start_date")
                end_date = data.get("end_date")
                if start_date and end_date:
                    await self.generate_custom_report(start_date, end_date, recipients)
    
    def configure_report(self, report_type: str, config: Dict[str, Any]) -> bool:
        """
        Configurar un informe periódico.
        
        Args:
            report_type: Tipo de informe (daily, weekly, monthly)
            config: Configuración del informe
            
        Returns:
            True si se configuró correctamente, False en caso contrario
        """
        if report_type not in self.configured_reports:
            self.logger.error(f"Tipo de informe no válido: {report_type}")
            return False
        
        # Actualizar configuración
        self.configured_reports[report_type].update(config)
        self.logger.info(f"Informe {report_type} configurado")
        return True
    
    def add_recipient(self, email: str) -> None:
        """
        Añadir un destinatario para todos los informes.
        
        Args:
            email: Dirección de correo electrónico
        """
        if email not in self.report_recipients:
            self.report_recipients.append(email)
            self.logger.info(f"Destinatario añadido: {email}")
    
    def remove_recipient(self, email: str) -> bool:
        """
        Eliminar un destinatario.
        
        Args:
            email: Dirección de correo electrónico
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        if email in self.report_recipients:
            self.report_recipients.remove(email)
            self.logger.info(f"Destinatario eliminado: {email}")
            return True
        return False
    
    async def generate_daily_report(self, recipients: Optional[List[str]] = None) -> str:
        """
        Generar informe diario.
        
        Args:
            recipients: Lista de destinatarios (opcional)
            
        Returns:
            Ruta al archivo del informe
        """
        # Obtener fecha
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        # Crear informe
        self.logger.info("Generando informe diario")
        
        # Generar datos para el informe
        report_data = await self._collect_daily_data(yesterday)
        
        # Generar archivo
        filename = f"daily_report_{yesterday.strftime('%Y%m%d')}.html"
        file_path = os.path.join(self.report_dir, filename)
        
        # Generar HTML
        html_content = self._generate_daily_html(report_data, yesterday)
        
        # Guardar archivo
        with open(file_path, "w") as f:
            f.write(html_content)
        
        # Enviar por correo si hay destinatarios
        if recipients:
            await self._send_report_email(
                recipients,
                f"Informe diario de trading - {yesterday.strftime('%d/%m/%Y')}",
                html_content,
                report_data["plots"]
            )
        
        self.logger.info(f"Informe diario generado: {file_path}")
        return file_path
    
    async def generate_weekly_report(self, recipients: Optional[List[str]] = None) -> str:
        """
        Generar informe semanal.
        
        Args:
            recipients: Lista de destinatarios (opcional)
            
        Returns:
            Ruta al archivo del informe
        """
        # Obtener fechas
        today = datetime.now()
        end_date = today - timedelta(days=1)  # Ayer
        start_date = end_date - timedelta(days=6)  # Hace una semana
        
        # Crear informe
        self.logger.info("Generando informe semanal")
        
        # Generar datos para el informe
        report_data = await self._collect_weekly_data(start_date, end_date)
        
        # Generar archivo
        filename = f"weekly_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.html"
        file_path = os.path.join(self.report_dir, filename)
        
        # Generar HTML
        html_content = self._generate_weekly_html(report_data, start_date, end_date)
        
        # Guardar archivo
        with open(file_path, "w") as f:
            f.write(html_content)
        
        # Enviar por correo si hay destinatarios
        if recipients:
            await self._send_report_email(
                recipients,
                f"Informe semanal de trading - {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}",
                html_content,
                report_data["plots"]
            )
        
        self.logger.info(f"Informe semanal generado: {file_path}")
        return file_path
    
    async def generate_monthly_report(self, recipients: Optional[List[str]] = None) -> str:
        """
        Generar informe mensual.
        
        Args:
            recipients: Lista de destinatarios (opcional)
            
        Returns:
            Ruta al archivo del informe
        """
        # Obtener fechas
        today = datetime.now()
        first_day_current_month = today.replace(day=1)
        last_day_prev_month = first_day_current_month - timedelta(days=1)
        first_day_prev_month = last_day_prev_month.replace(day=1)
        
        # Crear informe
        self.logger.info("Generando informe mensual")
        
        # Generar datos para el informe
        report_data = await self._collect_monthly_data(first_day_prev_month, last_day_prev_month)
        
        # Generar archivo
        filename = f"monthly_report_{first_day_prev_month.strftime('%Y%m')}.html"
        file_path = os.path.join(self.report_dir, filename)
        
        # Generar HTML
        html_content = self._generate_monthly_html(report_data, first_day_prev_month, last_day_prev_month)
        
        # Guardar archivo
        with open(file_path, "w") as f:
            f.write(html_content)
        
        # Enviar por correo si hay destinatarios
        if recipients:
            await self._send_report_email(
                recipients,
                f"Informe mensual de trading - {first_day_prev_month.strftime('%B %Y')}",
                html_content,
                report_data["plots"]
            )
        
        self.logger.info(f"Informe mensual generado: {file_path}")
        return file_path
    
    async def generate_custom_report(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime],
        recipients: Optional[List[str]] = None
    ) -> str:
        """
        Generar informe personalizado para un período específico.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            recipients: Lista de destinatarios (opcional)
            
        Returns:
            Ruta al archivo del informe
        """
        # Convertir fechas si son strings
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Crear informe
        self.logger.info(f"Generando informe personalizado: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
        
        # Generar datos para el informe
        report_data = await self._collect_custom_data(start_date, end_date)
        
        # Generar archivo
        filename = f"custom_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.html"
        file_path = os.path.join(self.report_dir, filename)
        
        # Generar HTML
        html_content = self._generate_custom_html(report_data, start_date, end_date)
        
        # Guardar archivo
        with open(file_path, "w") as f:
            f.write(html_content)
        
        # Enviar por correo si hay destinatarios
        if recipients:
            await self._send_report_email(
                recipients,
                f"Informe personalizado - {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}",
                html_content,
                report_data["plots"]
            )
        
        self.logger.info(f"Informe personalizado generado: {file_path}")
        return file_path
    
    async def generate_strategy_report(
        self, 
        strategy_name: str,
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime],
        recipients: Optional[List[str]] = None
    ) -> str:
        """
        Generar informe detallado para una estrategia específica.
        
        Args:
            strategy_name: Nombre de la estrategia
            start_date: Fecha de inicio
            end_date: Fecha de fin
            recipients: Lista de destinatarios (opcional)
            
        Returns:
            Ruta al archivo del informe
        """
        # Convertir fechas si son strings
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Crear informe
        self.logger.info(f"Generando informe de estrategia {strategy_name}")
        
        # Generar datos para el informe
        report_data = await self._collect_strategy_data(strategy_name, start_date, end_date)
        
        # Generar archivo
        filename = f"strategy_report_{strategy_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.html"
        file_path = os.path.join(self.report_dir, filename)
        
        # Generar HTML
        html_content = self._generate_strategy_html(report_data, strategy_name, start_date, end_date)
        
        # Guardar archivo
        with open(file_path, "w") as f:
            f.write(html_content)
        
        # Enviar por correo si hay destinatarios
        if recipients:
            await self._send_report_email(
                recipients,
                f"Informe de estrategia {strategy_name} - {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}",
                html_content,
                report_data["plots"]
            )
        
        self.logger.info(f"Informe de estrategia generado: {file_path}")
        return file_path
    
    async def _collect_daily_data(self, date: datetime) -> Dict[str, Any]:
        """
        Recopilar datos para el informe diario.
        
        Args:
            date: Fecha del informe
            
        Returns:
            Datos para el informe
        """
        # Este método recopila datos de trades, rendimiento, etc.
        # para la fecha específica desde la base de datos
        
        # Por ahora, generamos datos de ejemplo
        # En una implementación real, se consultaría la base de datos
        
        # Crear estructura de datos
        result = {
            "date": date.strftime("%Y-%m-%d"),
            "summary": {
                "total_trades": 0,
                "profitable_trades": 0,
                "loss_trades": 0,
                "total_profit": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0,
                "win_rate": 0.0
            },
            "trades": [],
            "strategies": {},
            "portfolio": {
                "starting_balance": 0.0,
                "ending_balance": 0.0,
                "profit_loss": 0.0,
                "profit_loss_percent": 0.0
            },
            "logs": {
                "total": 0,
                "by_level": {},
                "entries": []
            },
            "plots": {}
        }
        
        # Intentar recuperar datos de la base de datos o cache
        try:
            # Aquí se conectaría con el repositorio para obtener datos reales
            # Por ahora, generamos datos simulados
            # Esto simularía datos de trades del día
            trades = await self._simulate_trades_data(date, 15)  # 15 trades por día
            
            # Calcular métricas
            profitable_trades = [t for t in trades if t["profit"] > 0]
            loss_trades = [t for t in trades if t["profit"] <= 0]
            
            total_profit = sum(t["profit"] for t in trades)
            
            max_profit = max([t["profit"] for t in trades]) if trades else 0
            max_loss = min([t["profit"] for t in trades]) if trades else 0
            
            win_rate = len(profitable_trades) / len(trades) if trades else 0
            
            # Datos de estrategias
            strategies = await self._simulate_strategy_data(date)
            
            # Datos de portfolio
            starting_balance = 10000.0  # Simulado
            profit_loss_percent = total_profit / starting_balance
            ending_balance = starting_balance + total_profit
            
            # Actualizar resultado
            result["summary"] = {
                "total_trades": len(trades),
                "profitable_trades": len(profitable_trades),
                "loss_trades": len(loss_trades),
                "total_profit": total_profit,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "win_rate": win_rate
            }
            
            result["trades"] = trades
            result["strategies"] = strategies
            
            result["portfolio"] = {
                "starting_balance": starting_balance,
                "ending_balance": ending_balance,
                "profit_loss": total_profit,
                "profit_loss_percent": profit_loss_percent
            }
            
            # Generar gráficos
            result["plots"] = await self._generate_daily_plots(trades, strategies, date)
            
            # Obtener logs relevantes para el día
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
                
                # Actualizar la sección de logs en el resultado
                result["logs"] = {
                    "total": len(logs_data),
                    "by_level": log_stats.get("by_level", {}),
                    "by_component": log_stats.get("by_component", {}),
                    "entries": logs_data
                }
                
                self.logger.info(f"Se añadieron {len(logs_data)} registros de logs al informe diario.")
            except Exception as e:
                self.logger.error(f"Error al obtener logs para el informe diario: {e}")
            
        except Exception as e:
            self.logger.error(f"Error al recopilar datos diarios: {e}")
        
        return result
    
    async def _collect_weekly_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Recopilar datos para el informe semanal.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Datos para el informe
        """
        # Crear estructura de datos
        result = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "summary": {
                "total_trades": 0,
                "profitable_trades": 0,
                "loss_trades": 0,
                "total_profit": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0,
                "win_rate": 0.0,
                "average_trade": 0.0
            },
            "daily_summary": [],
            "trades": [],
            "strategies": {},
            "portfolio": {
                "starting_balance": 0.0,
                "ending_balance": 0.0,
                "profit_loss": 0.0,
                "profit_loss_percent": 0.0
            },
            "market_correlation": 0.0,
            "plots": {}
        }
        
        # Intentar recuperar datos
        try:
            # Simular datos de cada día del período
            daily_data = []
            all_trades = []
            
            current_date = start_date
            while current_date <= end_date:
                # Obtener datos diarios
                day_data = await self._collect_daily_data(current_date)
                daily_data.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "trades": len(day_data["trades"]),
                    "profit": day_data["summary"]["total_profit"],
                    "win_rate": day_data["summary"]["win_rate"]
                })
                
                # Acumular trades
                all_trades.extend(day_data["trades"])
                
                current_date += timedelta(days=1)
            
            # Calcular métricas
            profitable_trades = [t for t in all_trades if t["profit"] > 0]
            loss_trades = [t for t in all_trades if t["profit"] <= 0]
            
            total_profit = sum(t["profit"] for t in all_trades)
            
            max_profit = max([t["profit"] for t in all_trades]) if all_trades else 0
            max_loss = min([t["profit"] for t in all_trades]) if all_trades else 0
            
            win_rate = len(profitable_trades) / len(all_trades) if all_trades else 0
            average_trade = total_profit / len(all_trades) if all_trades else 0
            
            # Datos de estrategias (acumulado semanal)
            strategies = await self._simulate_strategy_data_period(start_date, end_date)
            
            # Datos de portfolio
            starting_balance = 10000.0  # Simulado
            profit_loss_percent = total_profit / starting_balance
            ending_balance = starting_balance + total_profit
            
            # Simulación de correlación con el mercado
            market_correlation = np.random.uniform(-1.0, 1.0)
            
            # Actualizar resultado
            result["summary"] = {
                "total_trades": len(all_trades),
                "profitable_trades": len(profitable_trades),
                "loss_trades": len(loss_trades),
                "total_profit": total_profit,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "win_rate": win_rate,
                "average_trade": average_trade
            }
            
            result["daily_summary"] = daily_data
            result["trades"] = all_trades
            result["strategies"] = strategies
            
            result["portfolio"] = {
                "starting_balance": starting_balance,
                "ending_balance": ending_balance,
                "profit_loss": total_profit,
                "profit_loss_percent": profit_loss_percent
            }
            
            result["market_correlation"] = market_correlation
            
            # Generar gráficos
            result["plots"] = await self._generate_weekly_plots(all_trades, daily_data, strategies, start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"Error al recopilar datos semanales: {e}")
        
        return result
    
    async def _collect_monthly_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Recopilar datos para el informe mensual.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Datos para el informe
        """
        # Similar a weekly_data pero con más métricas y análisis
        # Para simplificar, usamos la misma estructura pero añadimos más métricas
        
        # Obtener datos base (similar a semanal)
        base_data = await self._collect_weekly_data(start_date, end_date)
        
        # Añadir métricas adicionales para el informe mensual
        result = base_data
        
        # Métricas de riesgo
        result["risk"] = {
            "sharpe_ratio": np.random.uniform(0.5, 3.0),  # Simulado
            "sortino_ratio": np.random.uniform(0.7, 4.0),  # Simulado
            "max_drawdown": np.random.uniform(0.02, 0.15),  # Simulado
            "var_95": np.random.uniform(0.01, 0.05),  # Simulado
            "volatility": np.random.uniform(0.01, 0.08)  # Simulado
        }
        
        # Análisis por mercado
        result["markets"] = {}
        for market in ["crypto", "forex", "stocks"]:
            result["markets"][market] = {
                "trades": int(np.random.uniform(10, 100)),
                "win_rate": np.random.uniform(0.4, 0.7),
                "profit": np.random.uniform(-1000, 3000)
            }
        
        # Generar gráficos mensuales (más detallados)
        result["plots"].update(await self._generate_monthly_plots(result["trades"], result["daily_summary"], result["strategies"], start_date, end_date))
        
        return result
    
    async def _collect_custom_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Recopilar datos para un informe personalizado.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Datos para el informe
        """
        # Usar la estructura mensual para informes personalizados
        return await self._collect_monthly_data(start_date, end_date)
    
    async def _collect_strategy_data(self, strategy_name: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Recopilar datos para un informe de estrategia específica.
        
        Args:
            strategy_name: Nombre de la estrategia
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Datos para el informe
        """
        # Base de datos personalizados
        result = await self._collect_custom_data(start_date, end_date)
        
        # Filtrar solo trades de esta estrategia
        if "trades" in result:
            result["trades"] = [t for t in result["trades"] if t.get("strategy") == strategy_name]
        
        # Recalcular métricas
        trades = result.get("trades", [])
        profitable_trades = [t for t in trades if t["profit"] > 0]
        loss_trades = [t for t in trades if t["profit"] <= 0]
        
        total_profit = sum(t["profit"] for t in trades)
        
        max_profit = max([t["profit"] for t in trades]) if trades else 0
        max_loss = min([t["profit"] for t in trades]) if trades else 0
        
        win_rate = len(profitable_trades) / len(trades) if trades else 0
        average_trade = total_profit / len(trades) if trades else 0
        
        # Actualizar resultado
        result["summary"] = {
            "total_trades": len(trades),
            "profitable_trades": len(profitable_trades),
            "loss_trades": len(loss_trades),
            "total_profit": total_profit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "win_rate": win_rate,
            "average_trade": average_trade
        }
        
        # Información adicional de la estrategia
        result["strategy_info"] = {
            "name": strategy_name,
            "description": f"Estrategia basada en {strategy_name}",
            "parameters": {
                "param1": 10,
                "param2": 20
            },
            "performance_metrics": {
                "expectancy": (win_rate * max_profit) - ((1 - win_rate) * abs(max_loss)) if max_loss else 0,
                "profit_factor": sum(t["profit"] for t in profitable_trades) / abs(sum(t["profit"] for t in loss_trades)) if loss_trades and sum(t["profit"] for t in loss_trades) != 0 else 0,
                "recovery_factor": total_profit / abs(max_loss) if max_loss else 0
            },
            "optimization": {
                "best_params": {
                    "param1": 12,
                    "param2": 18
                },
                "improvement": "15%"
            }
        }
        
        # Generar gráficos específicos para la estrategia
        result["plots"] = await self._generate_strategy_plots(trades, strategy_name, start_date, end_date)
        
        return result
    
    async def _simulate_trades_data(self, date: datetime, num_trades: int = 10) -> List[Dict[str, Any]]:
        """
        Simular datos de trades para testing.
        
        Args:
            date: Fecha para los trades
            num_trades: Número de trades a generar
            
        Returns:
            Lista de trades simulados
        """
        trades = []
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"]
        strategies = ["ma_crossover", "rsi", "bollinger_bands", "macd", "sentiment"]
        
        for i in range(num_trades):
            # Determinar hora aleatoria
            hours = np.random.randint(0, 23)
            minutes = np.random.randint(0, 59)
            seconds = np.random.randint(0, 59)
            
            trade_time = date.replace(hour=hours, minute=minutes, second=seconds)
            
            # Determinar si es ganancia o pérdida (60% probabilidad de ganancia)
            is_profit = np.random.random() < 0.6
            
            # Generar monto de ganancia/pérdida
            amount = np.random.uniform(10, 500)
            if not is_profit:
                amount = -amount
            
            # Crear trade
            trade = {
                "id": f"trade_{date.strftime('%Y%m%d')}_{i}",
                "symbol": np.random.choice(symbols),
                "strategy": np.random.choice(strategies),
                "side": "buy" if np.random.random() < 0.5 else "sell",
                "open_time": trade_time.isoformat(),
                "close_time": (trade_time + timedelta(minutes=np.random.randint(5, 120))).isoformat(),
                "entry_price": np.random.uniform(100, 50000),
                "exit_price": 0,
                "amount": np.random.uniform(0.01, 2.0),
                "profit": amount,
                "profit_percent": amount / 10000  # Simplificación
            }
            
            # Calcular precio de salida basado en ganancia/pérdida
            if trade["side"] == "buy":
                trade["exit_price"] = trade["entry_price"] * (1 + trade["profit_percent"])
            else:
                trade["exit_price"] = trade["entry_price"] * (1 - trade["profit_percent"])
            
            trades.append(trade)
        
        return trades
    
    async def _simulate_strategy_data(self, date: datetime) -> Dict[str, Any]:
        """
        Simular datos de rendimiento de estrategias para testing.
        
        Args:
            date: Fecha para los datos
            
        Returns:
            Datos de estrategias simulados
        """
        strategies = {
            "ma_crossover": {
                "trades": np.random.randint(1, 10),
                "profit": np.random.uniform(-200, 800),
                "win_rate": np.random.uniform(0.4, 0.8)
            },
            "rsi": {
                "trades": np.random.randint(1, 10),
                "profit": np.random.uniform(-200, 800),
                "win_rate": np.random.uniform(0.4, 0.8)
            },
            "bollinger_bands": {
                "trades": np.random.randint(1, 10),
                "profit": np.random.uniform(-200, 800),
                "win_rate": np.random.uniform(0.4, 0.8)
            },
            "macd": {
                "trades": np.random.randint(1, 10),
                "profit": np.random.uniform(-200, 800),
                "win_rate": np.random.uniform(0.4, 0.8)
            },
            "sentiment": {
                "trades": np.random.randint(1, 10),
                "profit": np.random.uniform(-200, 800),
                "win_rate": np.random.uniform(0.4, 0.8)
            }
        }
        
        return strategies
    
    async def _simulate_strategy_data_period(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Simular datos de rendimiento de estrategias para un período.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Datos de estrategias simulados
        """
        # Similar a strategy_data pero con más trades acumulados
        strategies = await self._simulate_strategy_data(start_date)
        
        # Multiplicar por el número de días para simular acumulación
        days = (end_date - start_date).days + 1
        
        for strategy in strategies:
            strategies[strategy]["trades"] *= days
            strategies[strategy]["profit"] *= days
        
        return strategies
    
    async def _generate_daily_plots(
        self, 
        trades: List[Dict[str, Any]], 
        strategies: Dict[str, Any],
        date: datetime
    ) -> Dict[str, str]:
        """
        Generar gráficos para el informe diario.
        
        Args:
            trades: Lista de trades
            strategies: Datos de estrategias
            date: Fecha del informe
            
        Returns:
            Diccionario con rutas a los gráficos
        """
        plots = {}
        
        try:
            # 1. Gráfico de distribución de ganancias
            plt.figure(figsize=(10, 6))
            profits = [t["profit"] for t in trades]
            plt.hist(profits, bins=15, alpha=0.7, color='blue')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title(f"Distribución de ganancias - {date.strftime('%d/%m/%Y')}")
            plt.xlabel("Ganancia")
            plt.ylabel("Frecuencia")
            plt.grid(axis='y', alpha=0.75)
            
            profit_dist_path = f"{self.plot_dir}/profit_dist_{date.strftime('%Y%m%d')}.png"
            plt.savefig(profit_dist_path)
            plt.close()
            
            plots["profit_distribution"] = profit_dist_path
            
            # 2. Gráfico de rendimiento por estrategia
            plt.figure(figsize=(10, 6))
            strategy_names = list(strategies.keys())
            strategy_profits = [strategies[s]["profit"] for s in strategy_names]
            
            colors = ['green' if p > 0 else 'red' for p in strategy_profits]
            plt.bar(strategy_names, strategy_profits, color=colors)
            plt.title(f"Rendimiento por estrategia - {date.strftime('%d/%m/%Y')}")
            plt.xlabel("Estrategia")
            plt.ylabel("Ganancia")
            plt.grid(axis='y', alpha=0.75)
            plt.xticks(rotation=45)
            
            strategy_perf_path = f"{self.plot_dir}/strategy_perf_{date.strftime('%Y%m%d')}.png"
            plt.savefig(strategy_perf_path)
            plt.close()
            
            plots["strategy_performance"] = strategy_perf_path
            
            # 3. Gráfico de trades por hora
            plt.figure(figsize=(10, 6))
            trade_hours = [datetime.fromisoformat(t["open_time"]).hour for t in trades]
            plt.hist(trade_hours, bins=24, range=(0, 24), alpha=0.7, color='purple')
            plt.title(f"Trades por hora - {date.strftime('%d/%m/%Y')}")
            plt.xlabel("Hora del día")
            plt.ylabel("Número de trades")
            plt.grid(axis='y', alpha=0.75)
            plt.xticks(range(0, 24, 2))
            
            hourly_trades_path = f"{self.plot_dir}/hourly_trades_{date.strftime('%Y%m%d')}.png"
            plt.savefig(hourly_trades_path)
            plt.close()
            
            plots["hourly_trades"] = hourly_trades_path
            
        except Exception as e:
            self.logger.error(f"Error al generar gráficos diarios: {e}")
        
        return plots
    
    async def _generate_weekly_plots(
        self,
        trades: List[Dict[str, Any]],
        daily_summary: List[Dict[str, Any]],
        strategies: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, str]:
        """
        Generar gráficos para el informe semanal.
        
        Args:
            trades: Lista de trades
            daily_summary: Resumen diario
            strategies: Datos de estrategias
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Diccionario con rutas a los gráficos
        """
        plots = {}
        
        try:
            # 1. Gráfico de ganancias diarias
            plt.figure(figsize=(10, 6))
            dates = [d["date"] for d in daily_summary]
            profits = [d["profit"] for d in daily_summary]
            
            colors = ['green' if p > 0 else 'red' for p in profits]
            plt.bar(dates, profits, color=colors)
            plt.title(f"Ganancias diarias - {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
            plt.xlabel("Fecha")
            plt.ylabel("Ganancia")
            plt.grid(axis='y', alpha=0.75)
            plt.xticks(rotation=45)
            
            daily_profit_path = f"{self.plot_dir}/daily_profit_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
            plt.savefig(daily_profit_path)
            plt.close()
            
            plots["daily_profit"] = daily_profit_path
            
            # 2. Gráfico de rendimiento acumulado
            plt.figure(figsize=(10, 6))
            cumulative_profit = np.cumsum(profits)
            plt.plot(dates, cumulative_profit, marker='o', linestyle='-', color='blue')
            plt.title(f"Rendimiento acumulado - {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
            plt.xlabel("Fecha")
            plt.ylabel("Ganancia acumulada")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            cumulative_profit_path = f"{self.plot_dir}/cumulative_profit_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
            plt.savefig(cumulative_profit_path)
            plt.close()
            
            plots["cumulative_profit"] = cumulative_profit_path
            
            # 3. Gráfico de trades por par de trading
            plt.figure(figsize=(10, 6))
            symbols = [t["symbol"] for t in trades]
            symbol_counts = {}
            for symbol in symbols:
                if symbol in symbol_counts:
                    symbol_counts[symbol] += 1
                else:
                    symbol_counts[symbol] = 1
            
            symbol_names = list(symbol_counts.keys())
            symbol_values = [symbol_counts[s] for s in symbol_names]
            
            plt.pie(symbol_values, labels=symbol_names, autopct='%1.1f%%', startangle=90)
            plt.title(f"Trades por par - {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
            
            symbol_dist_path = f"{self.plot_dir}/symbol_dist_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
            plt.savefig(symbol_dist_path)
            plt.close()
            
            plots["symbol_distribution"] = symbol_dist_path
            
            # 4. Gráfico de rendimiento por estrategia
            plt.figure(figsize=(10, 6))
            strategy_names = list(strategies.keys())
            strategy_profits = [strategies[s]["profit"] for s in strategy_names]
            
            colors = ['green' if p > 0 else 'red' for p in strategy_profits]
            plt.bar(strategy_names, strategy_profits, color=colors)
            plt.title(f"Rendimiento por estrategia - {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
            plt.xlabel("Estrategia")
            plt.ylabel("Ganancia")
            plt.grid(axis='y', alpha=0.75)
            plt.xticks(rotation=45)
            
            strategy_perf_path = f"{self.plot_dir}/strategy_perf_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
            plt.savefig(strategy_perf_path)
            plt.close()
            
            plots["strategy_performance"] = strategy_perf_path
            
        except Exception as e:
            self.logger.error(f"Error al generar gráficos semanales: {e}")
        
        return plots
    
    async def _generate_monthly_plots(
        self,
        trades: List[Dict[str, Any]],
        daily_summary: List[Dict[str, Any]],
        strategies: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, str]:
        """
        Generar gráficos para el informe mensual.
        
        Args:
            trades: Lista de trades
            daily_summary: Resumen diario
            strategies: Datos de estrategias
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Diccionario con rutas a los gráficos
        """
        # Gráficos base (similares a semanales)
        plots = await self._generate_weekly_plots(trades, daily_summary, strategies, start_date, end_date)
        
        try:
            # Gráficos adicionales para el informe mensual
            
            # 1. Gráfico de distribución de duración de trades
            plt.figure(figsize=(10, 6))
            
            # Calcular duración en minutos
            durations = []
            for trade in trades:
                try:
                    open_time = datetime.fromisoformat(trade["open_time"])
                    close_time = datetime.fromisoformat(trade["close_time"])
                    duration_minutes = (close_time - open_time).total_seconds() / 60
                    durations.append(duration_minutes)
                except (ValueError, KeyError):
                    pass
            
            plt.hist(durations, bins=20, alpha=0.7, color='green')
            plt.title(f"Duración de trades - {start_date.strftime('%B %Y')}")
            plt.xlabel("Duración (minutos)")
            plt.ylabel("Frecuencia")
            plt.grid(True, alpha=0.3)
            
            duration_dist_path = f"{self.plot_dir}/duration_dist_{start_date.strftime('%Y%m')}.png"
            plt.savefig(duration_dist_path)
            plt.close()
            
            plots["duration_distribution"] = duration_dist_path
            
            # 2. Gráfico de análisis de drawdown
            plt.figure(figsize=(10, 6))
            
            # Calcular drawdown a partir de ganancias diarias
            profits = [d["profit"] for d in daily_summary]
            cumulative = np.cumsum(profits)
            rolling_max = np.maximum.accumulate(cumulative)
            drawdown = (rolling_max - cumulative) / (rolling_max + 1e-10)  # Evitar división por cero
            
            dates = [d["date"] for d in daily_summary]
            plt.plot(dates, drawdown * 100, color='red', linewidth=2)
            plt.title(f"Análisis de Drawdown - {start_date.strftime('%B %Y')}")
            plt.xlabel("Fecha")
            plt.ylabel("Drawdown (%)")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            drawdown_path = f"{self.plot_dir}/drawdown_{start_date.strftime('%Y%m')}.png"
            plt.savefig(drawdown_path)
            plt.close()
            
            plots["drawdown_analysis"] = drawdown_path
            
            # 3. Gráfico de matriz de correlación de estrategias
            strategy_names = list(strategies.keys())
            if len(strategy_names) >= 3:  # Necesitamos al menos 3 estrategias para una matriz útil
                plt.figure(figsize=(10, 8))
                
                # Crear datos simulados para correlación
                strategy_returns = {}
                for strategy in strategy_names:
                    # Generar returns diarios simulados para cada estrategia
                    strategy_returns[strategy] = np.random.normal(0.001, 0.01, 30)  # 30 días
                
                # Crear DataFrame
                returns_df = pd.DataFrame(strategy_returns)
                
                # Calcular matriz de correlación
                corr_matrix = returns_df.corr()
                
                # Visualizar
                plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar(label='Correlación')
                plt.xticks(np.arange(len(strategy_names)), strategy_names, rotation=45)
                plt.yticks(np.arange(len(strategy_names)), strategy_names)
                
                # Añadir valores
                for i in range(len(strategy_names)):
                    for j in range(len(strategy_names)):
                        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                                 ha="center", va="center", 
                                 color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                
                plt.title(f"Matriz de correlación de estrategias - {start_date.strftime('%B %Y')}")
                plt.tight_layout()
                
                corr_matrix_path = f"{self.plot_dir}/strategy_correlation_{start_date.strftime('%Y%m')}.png"
                plt.savefig(corr_matrix_path)
                plt.close()
                
                plots["strategy_correlation"] = corr_matrix_path
            
        except Exception as e:
            self.logger.error(f"Error al generar gráficos mensuales: {e}")
        
        return plots
    
    async def _generate_strategy_plots(
        self,
        trades: List[Dict[str, Any]],
        strategy_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, str]:
        """
        Generar gráficos para el informe de estrategia.
        
        Args:
            trades: Lista de trades de la estrategia
            strategy_name: Nombre de la estrategia
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Diccionario con rutas a los gráficos
        """
        plots = {}
        
        try:
            # 1. Gráfico de rendimiento acumulado
            plt.figure(figsize=(10, 6))
            
            # Ordenar trades por tiempo
            sorted_trades = sorted(trades, key=lambda x: x["open_time"])
            
            # Calcular rendimiento acumulado
            cumulative = np.cumsum([t["profit"] for t in sorted_trades])
            
            # Fechas para el eje X
            dates = [datetime.fromisoformat(t["open_time"]).strftime("%Y-%m-%d %H:%M") for t in sorted_trades]
            
            plt.plot(dates, cumulative, marker='o', linestyle='-', color='blue')
            plt.title(f"Rendimiento acumulado - Estrategia {strategy_name}")
            plt.xlabel("Fecha")
            plt.ylabel("Ganancia acumulada")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Añadir punto inicial en cero
            plt.plot([dates[0] if dates else start_date.strftime("%Y-%m-%d")], [0], marker='o', color='green')
            
            strategy_perf_path = f"{self.plot_dir}/strategy_{strategy_name}_perf_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
            plt.savefig(strategy_perf_path)
            plt.close()
            
            plots["strategy_performance"] = strategy_perf_path
            
            # 2. Gráfico de distribución de ganancias
            plt.figure(figsize=(10, 6))
            profits = [t["profit"] for t in trades]
            plt.hist(profits, bins=15, alpha=0.7, color='blue')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title(f"Distribución de ganancias - Estrategia {strategy_name}")
            plt.xlabel("Ganancia")
            plt.ylabel("Frecuencia")
            plt.grid(axis='y', alpha=0.75)
            
            profit_dist_path = f"{self.plot_dir}/strategy_{strategy_name}_profit_dist_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
            plt.savefig(profit_dist_path)
            plt.close()
            
            plots["profit_distribution"] = profit_dist_path
            
            # 3. Gráfico de rendimiento por par
            plt.figure(figsize=(10, 6))
            
            # Agrupar por par
            pair_profit = {}
            pair_count = {}
            
            for trade in trades:
                symbol = trade["symbol"]
                if symbol not in pair_profit:
                    pair_profit[symbol] = 0
                    pair_count[symbol] = 0
                
                pair_profit[symbol] += trade["profit"]
                pair_count[symbol] += 1
            
            # Ordenar por beneficio total
            pairs = sorted(pair_profit.keys(), key=lambda x: pair_profit[x], reverse=True)
            profits = [pair_profit[p] for p in pairs]
            
            colors = ['green' if p > 0 else 'red' for p in profits]
            plt.bar(pairs, profits, color=colors)
            plt.title(f"Rendimiento por par - Estrategia {strategy_name}")
            plt.xlabel("Par de trading")
            plt.ylabel("Ganancia")
            plt.grid(axis='y', alpha=0.75)
            plt.xticks(rotation=45)
            
            pair_perf_path = f"{self.plot_dir}/strategy_{strategy_name}_pair_perf_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
            plt.savefig(pair_perf_path)
            plt.close()
            
            plots["pair_performance"] = pair_perf_path
            
            # 4. Gráfico de consistencia (rendimiento por día de la semana)
            plt.figure(figsize=(10, 6))
            
            # Agrupar por día de la semana
            day_profit = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # Lunes a domingo
            day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            
            for trade in trades:
                try:
                    trade_date = datetime.fromisoformat(trade["open_time"])
                    day_of_week = trade_date.weekday()
                    day_profit[day_of_week] += trade["profit"]
                except (ValueError, KeyError):
                    pass
            
            # Ordenar por día de la semana
            profits = [day_profit[i] for i in range(7)]
            
            colors = ['green' if p > 0 else 'red' for p in profits]
            plt.bar(day_names, profits, color=colors)
            plt.title(f"Rendimiento por día - Estrategia {strategy_name}")
            plt.xlabel("Día de la semana")
            plt.ylabel("Ganancia")
            plt.grid(axis='y', alpha=0.75)
            
            day_perf_path = f"{self.plot_dir}/strategy_{strategy_name}_day_perf_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
            plt.savefig(day_perf_path)
            plt.close()
            
            plots["day_performance"] = day_perf_path
            
        except Exception as e:
            self.logger.error(f"Error al generar gráficos de estrategia: {e}")
        
        return plots
    
    def _generate_daily_html(self, data: Dict[str, Any], date: datetime) -> str:
        """
        Generar HTML para el informe diario.
        
        Args:
            data: Datos del informe
            date: Fecha del informe
            
        Returns:
            Contenido HTML
        """
        summary = data.get("summary", {})
        trades = data.get("trades", [])
        strategies = data.get("strategies", {})
        portfolio = data.get("portfolio", {})
        logs = data.get("logs", {"total": 0, "by_level": {}, "by_component": {}, "entries": []})
        
        # Generar HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Informe Diario de Trading - {date.strftime('%d/%m/%Y')}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .summary-box {{
                    display: inline-block;
                    width: 22%;
                    margin: 1%;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                .chart {{
                    width: 100%;
                    max-width: 600px;
                    margin: 20px auto;
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Informe Diario de Trading</h1>
                    <h2>{date.strftime('%d de %B, %Y')}</h2>
                </div>
                
                <div class="section">
                    <h2>Resumen</h2>
                    <div class="summary-box">
                        <h3>Trades</h3>
                        <p>{summary.get('total_trades', 0)}</p>
                    </div>
                    <div class="summary-box">
                        <h3>Win Rate</h3>
                        <p>{summary.get('win_rate', 0)*100:.1f}%</p>
                    </div>
                    <div class="summary-box">
                        <h3>Ganancia</h3>
                        <p class="{('positive' if summary.get('total_profit', 0) >= 0 else 'negative')}">${summary.get('total_profit', 0):.2f}</p>
                    </div>
                    <div class="summary-box">
                        <h3>Balance Final</h3>
                        <p>${portfolio.get('ending_balance', 0):.2f}</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Gráficos</h2>
        """
        
        # Añadir gráficos si existen
        for plot_name, plot_path in data.get("plots", {}).items():
            html += f"""
                    <div>
                        <h3>{plot_name.replace('_', ' ').title()}</h3>
                        <img src="{plot_path}" class="chart" alt="{plot_name}">
                    </div>
            """
        
        html += """
                </div>
                
                <div class="section">
                    <h2>Rendimiento por Estrategia</h2>
                    <table>
                        <tr>
                            <th>Estrategia</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Ganancia</th>
                        </tr>
        """
        
        # Añadir filas para cada estrategia
        for strategy_name, strategy_data in strategies.items():
            profit_class = "positive" if strategy_data.get("profit", 0) >= 0 else "negative"
            html += f"""
                        <tr>
                            <td>{strategy_name}</td>
                            <td>{strategy_data.get('trades', 0)}</td>
                            <td>{strategy_data.get('win_rate', 0)*100:.1f}%</td>
                            <td class="{profit_class}">${strategy_data.get('profit', 0):.2f}</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Trades del Día</h2>
                    <table>
                        <tr>
                            <th>ID</th>
                            <th>Par</th>
                            <th>Estrategia</th>
                            <th>Lado</th>
                            <th>Entrada</th>
                            <th>Salida</th>
                            <th>Ganancia</th>
                        </tr>
        """
        
        # Añadir filas para cada trade
        for trade in trades:
            profit_class = "positive" if trade.get("profit", 0) >= 0 else "negative"
            
            # Formatear horas
            try:
                open_time = datetime.fromisoformat(trade.get("open_time", "")).strftime("%H:%M:%S")
                close_time = datetime.fromisoformat(trade.get("close_time", "")).strftime("%H:%M:%S")
            except ValueError:
                open_time = "N/A"
                close_time = "N/A"
            
            html += f"""
                        <tr>
                            <td>{trade.get('id', 'N/A')}</td>
                            <td>{trade.get('symbol', 'N/A')}</td>
                            <td>{trade.get('strategy', 'N/A')}</td>
                            <td>{trade.get('side', 'N/A').upper()}</td>
                            <td>{open_time}</td>
                            <td>{close_time}</td>
                            <td class="{profit_class}">${trade.get('profit', 0):.2f} ({trade.get('profit_percent', 0)*100:.2f}%)</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Estadísticas de Portfolio</h2>
                    <table>
                        <tr>
                            <th>Balance Inicial</th>
                            <th>Balance Final</th>
                            <th>Ganancia</th>
                            <th>Rendimiento</th>
                        </tr>
                        <tr>
                            <td>${portfolio.get('starting_balance', 0):.2f}</td>
                            <td>${portfolio.get('ending_balance', 0):.2f}</td>
                            <td class="{('positive' if portfolio.get('profit_loss', 0) >= 0 else 'negative')}">${portfolio.get('profit_loss', 0):.2f}</td>
                            <td class="{('positive' if portfolio.get('profit_loss_percent', 0) >= 0 else 'negative')}">{portfolio.get('profit_loss_percent', 0)*100:.2f}%</td>
                        </tr>
                    </table>
                </div>
                
                <div class="footer">
                    <p>Generado por Sistema Genesis - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_weekly_html(self, data: Dict[str, Any], start_date: datetime, end_date: datetime) -> str:
        """
        Generar HTML para el informe semanal.
        
        Args:
            data: Datos del informe
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Contenido HTML
        """
        summary = data.get("summary", {})
        daily_summary = data.get("daily_summary", [])
        strategies = data.get("strategies", {})
        portfolio = data.get("portfolio", {})
        
        # Generar HTML (similar al diario pero con elementos adicionales)
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Informe Semanal de Trading - {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .summary-box {{
                    display: inline-block;
                    width: 22%;
                    margin: 1%;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                .chart {{
                    width: 100%;
                    max-width: 600px;
                    margin: 20px auto;
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Informe Semanal de Trading</h1>
                    <h2>{start_date.strftime('%d de %B')} - {end_date.strftime('%d de %B, %Y')}</h2>
                </div>
                
                <div class="section">
                    <h2>Resumen</h2>
                    <div class="summary-box">
                        <h3>Trades</h3>
                        <p>{summary.get('total_trades', 0)}</p>
                    </div>
                    <div class="summary-box">
                        <h3>Win Rate</h3>
                        <p>{summary.get('win_rate', 0)*100:.1f}%</p>
                    </div>
                    <div class="summary-box">
                        <h3>Ganancia</h3>
                        <p class="{('positive' if summary.get('total_profit', 0) >= 0 else 'negative')}">${summary.get('total_profit', 0):.2f}</p>
                    </div>
                    <div class="summary-box">
                        <h3>Rendimiento</h3>
                        <p class="{('positive' if portfolio.get('profit_loss_percent', 0) >= 0 else 'negative')}">{portfolio.get('profit_loss_percent', 0)*100:.2f}%</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Gráficos</h2>
        """
        
        # Añadir gráficos si existen
        for plot_name, plot_path in data.get("plots", {}).items():
            html += f"""
                    <div>
                        <h3>{plot_name.replace('_', ' ').title()}</h3>
                        <img src="{plot_path}" class="chart" alt="{plot_name}">
                    </div>
            """
        
        html += """
                </div>
                
                <div class="section">
                    <h2>Resumen Diario</h2>
                    <table>
                        <tr>
                            <th>Fecha</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Ganancia</th>
                        </tr>
        """
        
        # Añadir filas para cada día
        for day in daily_summary:
            profit_class = "positive" if day.get("profit", 0) >= 0 else "negative"
            html += f"""
                        <tr>
                            <td>{day.get('date', 'N/A')}</td>
                            <td>{day.get('trades', 0)}</td>
                            <td>{day.get('win_rate', 0)*100:.1f}%</td>
                            <td class="{profit_class}">${day.get('profit', 0):.2f}</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Rendimiento por Estrategia</h2>
                    <table>
                        <tr>
                            <th>Estrategia</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Ganancia</th>
                        </tr>
        """
        
        # Añadir filas para cada estrategia
        for strategy_name, strategy_data in strategies.items():
            profit_class = "positive" if strategy_data.get("profit", 0) >= 0 else "negative"
            html += f"""
                        <tr>
                            <td>{strategy_name}</td>
                            <td>{strategy_data.get('trades', 0)}</td>
                            <td>{strategy_data.get('win_rate', 0)*100:.1f}%</td>
                            <td class="{profit_class}">${strategy_data.get('profit', 0):.2f}</td>
                        </tr>
            """
        
        html += f"""
                    </table>
                </div>
                
                <div class="section">
                    <h2>Estadísticas Adicionales</h2>
                    <table>
                        <tr>
                            <th>Métrica</th>
                            <th>Valor</th>
                        </tr>
                        <tr>
                            <td>Trade Promedio</td>
                            <td class="{('positive' if summary.get('average_trade', 0) >= 0 else 'negative')}">${summary.get('average_trade', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Máxima Ganancia</td>
                            <td class="positive">${summary.get('max_profit', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Máxima Pérdida</td>
                            <td class="negative">${summary.get('max_loss', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Correlación con el Mercado</td>
                            <td>{data.get('market_correlation', 0):.2f}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="footer">
                    <p>Generado por Sistema Genesis - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_monthly_html(self, data: Dict[str, Any], start_date: datetime, end_date: datetime) -> str:
        """
        Generar HTML para el informe mensual.
        
        Args:
            data: Datos del informe
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Contenido HTML
        """
        # Estructura similar a la semanal pero con secciones adicionales
        html = self._generate_weekly_html(data, start_date, end_date)
        
        # Reemplazar el título
        html = html.replace("<h1>Informe Semanal de Trading</h1>", "<h1>Informe Mensual de Trading</h1>")
        
        # Añadir secciones adicionales antes del footer
        risk_html = ""
        if "risk" in data:
            risk = data["risk"]
            risk_html = f"""
                <div class="section">
                    <h2>Métricas de Riesgo</h2>
                    <table>
                        <tr>
                            <th>Métrica</th>
                            <th>Valor</th>
                            <th>Interpretación</th>
                        </tr>
                        <tr>
                            <td>Ratio Sharpe</td>
                            <td>{risk.get('sharpe_ratio', 0):.2f}</td>
                            <td>{'Excelente' if risk.get('sharpe_ratio', 0) > 2 else 'Bueno' if risk.get('sharpe_ratio', 0) > 1 else 'Regular' if risk.get('sharpe_ratio', 0) > 0.5 else 'Pobre'}</td>
                        </tr>
                        <tr>
                            <td>Ratio Sortino</td>
                            <td>{risk.get('sortino_ratio', 0):.2f}</td>
                            <td>{'Excelente' if risk.get('sortino_ratio', 0) > 2.5 else 'Bueno' if risk.get('sortino_ratio', 0) > 1.5 else 'Regular' if risk.get('sortino_ratio', 0) > 0.8 else 'Pobre'}</td>
                        </tr>
                        <tr>
                            <td>Máximo Drawdown</td>
                            <td>{risk.get('max_drawdown', 0)*100:.2f}%</td>
                            <td>{'Bajo' if risk.get('max_drawdown', 0) < 0.05 else 'Moderado' if risk.get('max_drawdown', 0) < 0.10 else 'Alto' if risk.get('max_drawdown', 0) < 0.20 else 'Muy Alto'}</td>
                        </tr>
                        <tr>
                            <td>VaR (95%)</td>
                            <td>{risk.get('var_95', 0)*100:.2f}%</td>
                            <td>Pérdida máxima esperada con 95% de confianza</td>
                        </tr>
                        <tr>
                            <td>Volatilidad</td>
                            <td>{risk.get('volatility', 0)*100:.2f}%</td>
                            <td>{'Baja' if risk.get('volatility', 0) < 0.02 else 'Moderada' if risk.get('volatility', 0) < 0.05 else 'Alta' if risk.get('volatility', 0) < 0.10 else 'Muy Alta'}</td>
                        </tr>
                    </table>
                </div>
            """
        
        markets_html = ""
        if "markets" in data:
            markets = data["markets"]
            markets_html = f"""
                <div class="section">
                    <h2>Análisis por Mercado</h2>
                    <table>
                        <tr>
                            <th>Mercado</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Ganancia</th>
                        </tr>
            """
            
            for market_name, market_data in markets.items():
                profit_class = "positive" if market_data.get("profit", 0) >= 0 else "negative"
                markets_html += f"""
                        <tr>
                            <td>{market_name.capitalize()}</td>
                            <td>{market_data.get('trades', 0)}</td>
                            <td>{market_data.get('win_rate', 0)*100:.1f}%</td>
                            <td class="{profit_class}">${market_data.get('profit', 0):.2f}</td>
                        </tr>
                """
            
            markets_html += """
                    </table>
                </div>
            """
        
        # Insertar secciones antes del footer
        footer_index = html.find("<div class=\"footer\">")
        if footer_index != -1:
            html = html[:footer_index] + risk_html + markets_html + html[footer_index:]
        
        return html
    
    def _generate_custom_html(self, data: Dict[str, Any], start_date: datetime, end_date: datetime) -> str:
        """
        Generar HTML para un informe personalizado.
        
        Args:
            data: Datos del informe
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Contenido HTML
        """
        # Usar la estructura del informe mensual
        html = self._generate_monthly_html(data, start_date, end_date)
        
        # Reemplazar el título
        html = html.replace("<h1>Informe Mensual de Trading</h1>", 
                           f"<h1>Informe Personalizado de Trading</h1>\n<h3>Período: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}</h3>")
        
        return html
    
    def _generate_strategy_html(self, data: Dict[str, Any], strategy_name: str, start_date: datetime, end_date: datetime) -> str:
        """
        Generar HTML para un informe de estrategia.
        
        Args:
            data: Datos del informe
            strategy_name: Nombre de la estrategia
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Contenido HTML
        """
        # Generar HTML base
        summary = data.get("summary", {})
        trades = data.get("trades", [])
        portfolio = data.get("portfolio", {})
        strategy_info = data.get("strategy_info", {})
        
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Informe de Estrategia {strategy_name} - {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .summary-box {{
                    display: inline-block;
                    width: 22%;
                    margin: 1%;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                .chart {{
                    width: 100%;
                    max-width: 600px;
                    margin: 20px auto;
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Informe de Estrategia</h1>
                    <h2>{strategy_name}</h2>
                    <h3>{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}</h3>
                </div>
                
                <div class="section">
                    <h2>Información de la Estrategia</h2>
                    <p>{strategy_info.get('description', 'No hay descripción disponible.')}</p>
                    
                    <h3>Parámetros</h3>
                    <table>
                        <tr>
                            <th>Parámetro</th>
                            <th>Valor</th>
                        </tr>
        """
        
        # Añadir parámetros
        for param_name, param_value in strategy_info.get("parameters", {}).items():
            html += f"""
                        <tr>
                            <td>{param_name}</td>
                            <td>{param_value}</td>
                        </tr>
            """
        
        html += f"""
                    </table>
                </div>
                
                <div class="section">
                    <h2>Resumen de Rendimiento</h2>
                    <div class="summary-box">
                        <h3>Trades</h3>
                        <p>{summary.get('total_trades', 0)}</p>
                    </div>
                    <div class="summary-box">
                        <h3>Win Rate</h3>
                        <p>{summary.get('win_rate', 0)*100:.1f}%</p>
                    </div>
                    <div class="summary-box">
                        <h3>Ganancia</h3>
                        <p class="{('positive' if summary.get('total_profit', 0) >= 0 else 'negative')}">${summary.get('total_profit', 0):.2f}</p>
                    </div>
                    <div class="summary-box">
                        <h3>Trade Promedio</h3>
                        <p class="{('positive' if summary.get('average_trade', 0) >= 0 else 'negative')}">${summary.get('average_trade', 0):.2f}</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Métricas de Rendimiento</h2>
                    <table>
                        <tr>
                            <th>Métrica</th>
                            <th>Valor</th>
                        </tr>
                        <tr>
                            <td>Expectativa</td>
                            <td class="{('positive' if strategy_info.get('performance_metrics', {}).get('expectancy', 0) >= 0 else 'negative')}">${strategy_info.get('performance_metrics', {}).get('expectancy', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Factor de Beneficio</td>
                            <td>{strategy_info.get('performance_metrics', {}).get('profit_factor', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Factor de Recuperación</td>
                            <td>{strategy_info.get('performance_metrics', {}).get('recovery_factor', 0):.2f}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Gráficos</h2>
        """
        
        # Añadir gráficos si existen
        for plot_name, plot_path in data.get("plots", {}).items():
            html += f"""
                    <div>
                        <h3>{plot_name.replace('_', ' ').title()}</h3>
                        <img src="{plot_path}" class="chart" alt="{plot_name}">
                    </div>
            """
        
        html += """
                </div>
                
                <div class="section">
                    <h2>Trades Recientes</h2>
                    <table>
                        <tr>
                            <th>ID</th>
                            <th>Par</th>
                            <th>Lado</th>
                            <th>Entrada</th>
                            <th>Salida</th>
                            <th>Ganancia</th>
                        </tr>
        """
        
        # Añadir filas para cada trade (mostrar los últimos 10)
        for trade in trades[-10:]:
            profit_class = "positive" if trade.get("profit", 0) >= 0 else "negative"
            
            # Formatear horas
            try:
                open_time = datetime.fromisoformat(trade.get("open_time", "")).strftime("%d/%m/%Y %H:%M")
                close_time = datetime.fromisoformat(trade.get("close_time", "")).strftime("%d/%m/%Y %H:%M")
            except ValueError:
                open_time = "N/A"
                close_time = "N/A"
            
            html += f"""
                        <tr>
                            <td>{trade.get('id', 'N/A')}</td>
                            <td>{trade.get('symbol', 'N/A')}</td>
                            <td>{trade.get('side', 'N/A').upper()}</td>
                            <td>{open_time}</td>
                            <td>{close_time}</td>
                            <td class="{profit_class}">${trade.get('profit', 0):.2f} ({trade.get('profit_percent', 0)*100:.2f}%)</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
        """
        
        # Sección de optimización si está disponible
        if "optimization" in strategy_info:
            html += f"""
                <div class="section">
                    <h2>Optimización</h2>
                    <p>Mejora potencial: {strategy_info.get('optimization', {}).get('improvement', 'N/A')}</p>
                    
                    <h3>Parámetros Óptimos</h3>
                    <table>
                        <tr>
                            <th>Parámetro</th>
                            <th>Valor Original</th>
                            <th>Valor Óptimo</th>
                        </tr>
            """
            
            # Añadir filas para cada parámetro
            for param_name, param_value in strategy_info.get("parameters", {}).items():
                optimal_value = strategy_info.get("optimization", {}).get("best_params", {}).get(param_name, "N/A")
                html += f"""
                        <tr>
                            <td>{param_name}</td>
                            <td>{param_value}</td>
                            <td>{optimal_value}</td>
                        </tr>
                """
            
            html += """
                    </table>
                </div>
            """
        
        html += """
                <div class="footer">
                    <p>Generado por Sistema Genesis - </div></div>
        """
        
        return html
    
    async def _send_report_email(
        self, 
        recipients: List[str], 
        subject: str, 
        html_content: str,
        plots: Dict[str, str] = None
    ) -> bool:
        """
        Enviar informe por correo electrónico.
        
        Args:
            recipients: Lista de destinatarios
            subject: Asunto del correo
            html_content: Contenido HTML del informe
            plots: Diccionario de rutas a gráficos (opcional)
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        try:
            # Verificar que el notificador de email esté disponible
            if email_notifier is None:
                self.logger.error("Notificador de email no disponible")
                return False
            
            # Enviar a cada destinatario
            for recipient in recipients:
                # Crear mensaje de correo
                message = f"""
                Adjunto encontrará el informe de trading generado por el sistema Genesis.
                
                Este mensaje ha sido generado automáticamente, no responda a este correo.
                """
                
                # Enviar email con el notificador
                success = await email_notifier.send(
                    recipient=recipient,
                    subject=subject,
                    message=message,
                    html_message=html_content
                )
                
                if success:
                    self.logger.info(f"Informe enviado a {recipient}")
                else:
                    self.logger.error(f"Error al enviar informe a {recipient}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error al enviar informe por email: {e}")
            return False


# Exportación para uso fácil
report_generator = ReportGenerator()