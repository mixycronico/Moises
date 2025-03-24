"""
Buddha Trader: Estrategia de trading trascendental potenciada por Buddha AI.

Esta estrategia combina el enfoque humano (análisis técnico y gestión de riesgo dinámica)
con la sabiduría de Buddha AI para obtener una visión más profunda del mercado, 
mejorar la selección de activos y optimizar la gestión de riesgo.

Características principales:
- Análisis de mercado avanzado con Buddha para mejorar entradas
- Evaluación de riesgo potenciada por IA
- Detección de oportunidades con mayor precisión
- Análisis de sentimiento para evitar operaciones durante noticias negativas
"""

import ccxt
import talib
import numpy as np
import random
import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Importar el integrador Buddha
from genesis.trading.buddha_integrator import BuddhaIntegrator

# Configuración de logging
logger = logging.getLogger("genesis.strategies.buddha_trader")

class BuddhaTrader:
    """
    Estrategia de trading trascendental potenciada por Buddha AI.
    
    Esta estrategia combina indicadores técnicos (EMA, RSI, ADX) con el análisis
    avanzado de Buddha para tomar decisiones más precisas, gestionar el riesgo
    de forma dinámica y detectar oportunidades que los indicadores tradicionales
    podrían pasar por alto.
    """
    
    def __init__(self, 
                 capital: float = 150, 
                 emergency_fund: float = 20, 
                 next_cycle: float = 20, 
                 personal_use: float = 10,
                 buddha_config_path: str = "buddha_config.json"):
        """
        Inicializar el Buddha Trader.
        
        Args:
            capital: Capital inicial para trading
            emergency_fund: Fondo de emergencia
            next_cycle: Ahorro para próximo ciclo
            personal_use: Reserva para uso personal
            buddha_config_path: Ruta al archivo de configuración de Buddha
        """
        self.capital = capital
        self.reserve = 0
        self.emergency_fund = emergency_fund
        self.next_cycle = next_cycle
        self.personal_use = personal_use
        
        # Conexiones a exchanges
        self.exchanges = {
            "binance": self._create_exchange("binance"),
            "kucoin": self._create_exchange("kucoin"),
            "bybit": self._create_exchange("bybit")
        }
        self.current_exchange = "binance"  # Exchange activo por defecto
        
        # Historial para análisis
        self.atr_history = []  # Para ajuste dinámico de riesgo
        self.trade_history = []  # Para análisis de rendimiento
        
        # Inicializar Buddha AI
        self.buddha = BuddhaIntegrator(buddha_config_path)
        self.buddha_enabled = self.buddha.is_enabled()
        
        # Estado del trader
        self.active_trades = {}
        self.pending_orders = {}
        self.cycle_complete = False
        self.metrics = {
            "trades_total": 0,
            "trades_success": 0,
            "trades_failed": 0,
            "profit_total": 0.0,
            "max_drawdown": 0.0,
            "success_rate": 0.0,
            "avg_profit_per_trade": 0.0,
            "buddha_insights_used": 0
        }
        
        logger.info(f"Buddha Trader inicializado con capital ${capital:.2f}")
        logger.info(f"Buddha AI {'habilitado' if self.buddha_enabled else 'deshabilitado'}")
    
    def _create_exchange(self, exchange_id: str):
        """
        Crear conexión a exchange.
        
        Args:
            exchange_id: ID del exchange (binance, kucoin, bybit)
            
        Returns:
            Objeto de exchange
        """
        # En un entorno real, aquí se configurarían las API keys
        return getattr(ccxt, exchange_id)({
            "apiKey": "",  # API key del exchange
            "secret": "",  # Secret del exchange
            "enableRateLimit": True,  # Evitar ban por exceso de peticiones
        })
    
    def get_symbols(self) -> List[str]:
        """
        Obtener lista de símbolos para operar.
        
        Returns:
            Lista de símbolos
        """
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"]
    
    def rotate_exchange(self) -> str:
        """
        Rotar al siguiente exchange para distribuir operaciones.
        
        Returns:
            Nombre del exchange actual
        """
        exchanges = list(self.exchanges.keys())
        current_index = exchanges.index(self.current_exchange)
        next_index = (current_index + 1) % len(exchanges)
        self.current_exchange = exchanges[next_index]
        return self.current_exchange
    
    async def fetch_data(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> np.ndarray:
        """
        Obtener datos históricos de un símbolo.
        
        Args:
            symbol: Par de trading (ej. BTC/USDT)
            timeframe: Intervalo de tiempo (1m, 5m, 15m, 1h, etc.)
            limit: Número de velas a obtener
            
        Returns:
            Array con precios de cierre
        """
        try:
            # Usar el exchange actual
            exchange = self.exchanges[self.current_exchange]
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit)
            
            # Extraer precios de cierre (índice 4)
            close_prices = np.array([c[4] for c in ohlcv])
            
            # Extraer volúmenes para análisis (índice 5)
            volumes = np.array([c[5] for c in ohlcv])
            
            # Calcular ATR actual
            high = np.array([c[2] for c in ohlcv])
            low = np.array([c[3] for c in ohlcv])
            atr = self._calculate_atr(high, low, close_prices)
            
            # Guardar ATR actual para ajuste dinámico
            if len(atr) > 0:
                last_atr = atr[-1]
                self.atr_history.append(last_atr)
                if len(self.atr_history) > 5:
                    self.atr_history.pop(0)
            
            return {
                "close": close_prices,
                "high": high,
                "low": low,
                "volume": volumes,
                "atr": atr
            }
        except Exception as e:
            logger.error(f"Error al obtener datos de {symbol}: {str(e)}")
            # Devolver datos vacíos en caso de error
            return {
                "close": np.array([]),
                "high": np.array([]),
                "low": np.array([]),
                "volume": np.array([]),
                "atr": np.array([])
            }
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calcular Average True Range (ATR).
        
        Args:
            high: Precios máximos
            low: Precios mínimos
            close: Precios de cierre
            period: Período para ATR
            
        Returns:
            Array con valores ATR
        """
        try:
            return talib.ATR(high, low, close, timeperiod=period)
        except Exception as e:
            logger.error(f"Error al calcular ATR: {str(e)}")
            return np.array([])
    
    async def get_technical_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener señales técnicas tradicionales.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Diccionario con señales técnicas
        """
        data = await self.fetch_data(symbol)
        
        # Verificar si hay datos válidos
        if len(data["close"]) == 0:
            return {"valid": False}
        
        close = data["close"]
        
        try:
            # Calcular indicadores
            ema9 = talib.EMA(close, timeperiod=9)
            ema21 = talib.EMA(close, timeperiod=21)
            rsi = talib.RSI(close, timeperiod=14)
            adx = talib.ADX(data["high"], data["low"], close, timeperiod=14)
            
            # Obtener valores actuales (último elemento)
            current_price = close[-1]
            current_ema9 = ema9[-1] if len(ema9) > 0 and not np.isnan(ema9[-1]) else current_price
            current_ema21 = ema21[-1] if len(ema21) > 0 and not np.isnan(ema21[-1]) else current_price
            current_rsi = rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50
            current_adx = adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 0
            
            # Volumen y ATR
            volume = data["volume"][-5:].mean() if len(data["volume"]) >= 5 else 0
            last_atr = data["atr"][-1] if len(data["atr"]) > 0 and not np.isnan(data["atr"][-1]) else 0
            
            # Condiciones para señales
            buy_signal = (
                current_ema9 > current_ema21 and  # EMA cruzada alcista
                50 < current_rsi < 70 and          # RSI en zona de compra pero no sobrecompra
                current_adx > 30                   # Tendencia fuerte
            )
            
            sell_signal = (
                current_ema9 < current_ema21 and  # EMA cruzada bajista
                current_rsi > 70 and              # RSI en zona de sobrecompra
                current_adx > 25                  # Tendencia definida
            )
            
            # Construir resultado
            return {
                "valid": True,
                "buy": buy_signal,
                "sell": sell_signal,
                "price": current_price,
                "ema9": current_ema9,
                "ema21": current_ema21,
                "rsi": current_rsi,
                "adx": current_adx,
                "volume": volume,
                "atr": last_atr
            }
        except Exception as e:
            logger.error(f"Error al calcular señales técnicas para {symbol}: {str(e)}")
            return {"valid": False}
    
    async def get_buddha_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener análisis avanzado de Buddha para el símbolo.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Análisis de Buddha o None si no está disponible
        """
        if not self.buddha_enabled:
            return None
        
        try:
            # Convertir símbolo de trading (BTC/USDT) a nombre de activo (Bitcoin)
            asset = symbol.split('/')[0]
            asset_name_map = {
                "BTC": "Bitcoin",
                "ETH": "Ethereum",
                "SOL": "Solana",
                "XRP": "Ripple",
                "ADA": "Cardano"
            }
            asset_name = asset_name_map.get(asset, asset)
            
            # Obtener análisis de mercado
            variables = ["precio", "volumen", "tendencia", "sentimiento", "noticias"]
            analysis = await self.buddha.analyze_market(asset_name, variables, timeframe=24)
            
            # Incrementar contador de uso de insights
            self.metrics["buddha_insights_used"] += 1
            
            return analysis
        except Exception as e:
            logger.error(f"Error al obtener análisis de Buddha para {symbol}: {str(e)}")
            return None
    
    async def calculate_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Calcular parámetros de operación integrando señales técnicas y Buddha.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Detalles de la operación o señal de espera
        """
        # Obtener señales técnicas
        signals = await self.get_technical_signals(symbol)
        
        # Verificar validez de señales
        if not signals.get("valid", False) or not signals.get("buy", False):
            return {"action": "wait"}
        
        # Obtener análisis de Buddha si está disponible
        buddha_analysis = await self.get_buddha_analysis(symbol)
        
        # Extraer información relevante
        price = signals["price"]
        atr = signals["atr"]
        
        # Calcular riesgo dinámico
        avg_atr = np.mean(self.atr_history) if self.atr_history else atr
        
        # Si Buddha está disponible, ajustar riesgo según su análisis
        risk_modifier = 1.0
        if buddha_analysis:
            # Ajustar según confianza y predicción de Buddha
            confidence = buddha_analysis.get("confidence", 0.8)
            predicted_trend = buddha_analysis.get("predicted_trend", "neutral")
            
            if predicted_trend == "bullish" and confidence > 0.8:
                risk_modifier = 1.2  # Aumentar riesgo en tendencias alcistas fuertes
            elif predicted_trend == "bearish" and confidence > 0.7:
                return {"action": "wait"}  # Evitar entrada en tendencias bajistas
            elif predicted_trend == "neutral":
                risk_modifier = 0.8  # Reducir riesgo en mercados sin dirección clara
        
        # Determinar riesgo base según volatilidad (ATR)
        base_risk = 0.03 if atr < 2 * avg_atr else 0.015  # Riesgo dinámico
        risk = base_risk * risk_modifier  # Aplicar modificador de Buddha
        
        # Limitar riesgo a un máximo del 3%
        risk = min(risk, 0.03)
        
        # Calcular tamaño de posición
        risk_amount = self.capital * risk
        position_size = risk_amount / atr if atr > 0 else 0
        
        # Establecer stop loss y take profit
        stop_loss = price - atr
        
        # Si Buddha está disponible, usar su predicción para take profit
        if buddha_analysis and "predicted_change_pct" in buddha_analysis:
            predicted_change = buddha_analysis["predicted_change_pct"] / 100
            take_profit = price * (1 + max(predicted_change, 0.05))  # Mínimo 5%
        else:
            take_profit = price * 1.06  # Default 6%
        
        # Calcular fees y slippage
        fee_rate = 0.001  # 0.1%
        slippage_rate = 0.005 if "SOL" in symbol or "ADA" in symbol or "XRP" in symbol else 0.002  # 0.5% altcoins, 0.2% BTC/ETH
        fee = price * position_size * fee_rate
        slippage = price * position_size * slippage_rate
        
        return {
            "action": "buy",
            "symbol": symbol,
            "amount": position_size,
            "entry": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "fee": fee,
            "slippage": slippage,
            "risk_pct": risk * 100,
            "buddha_insight": True if buddha_analysis else False
        }
    
    async def execute_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simular ejecución de operación en el entorno actual.
        
        En un entorno de producción, esta función enviaría la orden al exchange.
        Para esta simulación, solicita confirmación al usuario y registra el resultado.
        
        Args:
            trade: Detalles de la operación a ejecutar
            
        Returns:
            Resultado de la operación
        """
        if trade["action"] != "buy":
            return {"success": False, "reason": "No es una señal de compra"}
        
        # Mostrar detalles de la operación
        print(f"\n=== SEÑAL DE TRADING (Buddha: {'✓' if trade['buddha_insight'] else '✗'}) ===")
        print(f"Símbolo: {trade['symbol']} en {self.current_exchange}")
        print(f"Comprar {trade['amount']:.4f} a ${trade['entry']:.2f}")
        print(f"Stop Loss: ${trade['stop_loss']:.2f} (${trade['entry'] - trade['stop_loss']:.2f} riesgo)")
        print(f"Take Profit: ${trade['take_profit']:.2f} (${trade['take_profit'] - trade['entry']:.2f} beneficio)")
        print(f"Fee estimado: ${trade['fee']:.2f}, Slippage: ${trade['slippage']:.2f}")
        print(f"Riesgo: {trade['risk_pct']:.2f}% del capital (${self.capital * trade['risk_pct'] / 100:.2f})")
        
        # Solicitar confirmación del usuario
        confirmation = input("\n¿Ejecutar esta operación? (s/n): ").lower()
        if confirmation != "s":
            return {"success": False, "reason": "Operación cancelada por el usuario"}
        
        # Simular resultado (en producción, esto sería una orden real)
        trade_id = f"trade_{int(time.time())}_{random.randint(1000, 9999)}"
        self.active_trades[trade_id] = {
            **trade,
            "timestamp": time.time(),
            "exchange": self.current_exchange
        }
        
        # Solicitar precio de salida (simulación)
        try:
            outcome_price = float(input("\nIngresa el precio de salida (simulación): "))
            
            # Calcular resultado
            entry_cost = trade["entry"] * trade["amount"]
            exit_value = outcome_price * trade["amount"]
            costs = trade["fee"] + trade["slippage"]
            profit_loss = exit_value - entry_cost - costs
            profit_pct = (profit_loss / self.capital) * 100
            
            # Actualizar capital
            self.capital += profit_loss
            
            # Registrar resultado
            result = {
                "trade_id": trade_id,
                "symbol": trade["symbol"],
                "entry": trade["entry"],
                "exit": outcome_price,
                "amount": trade["amount"],
                "costs": costs,
                "profit_loss": profit_loss,
                "profit_pct": profit_pct,
                "success": profit_loss > 0,
                "buddha_insight": trade["buddha_insight"]
            }
            
            # Actualizar métricas
            self.metrics["trades_total"] += 1
            if profit_loss > 0:
                self.metrics["trades_success"] += 1
            else:
                self.metrics["trades_failed"] += 1
            self.metrics["profit_total"] += profit_loss
            self.metrics["success_rate"] = (self.metrics["trades_success"] / self.metrics["trades_total"]) * 100 if self.metrics["trades_total"] > 0 else 0
            self.metrics["avg_profit_per_trade"] = self.metrics["profit_total"] / self.metrics["trades_total"] if self.metrics["trades_total"] > 0 else 0
            
            # Eliminar de trades activos
            del self.active_trades[trade_id]
            
            # Agregar a historial
            self.trade_history.append(result)
            
            # Mostrar resultado
            print(f"\n=== RESULTADO DE LA OPERACIÓN ===")
            print(f"Beneficio/Pérdida: ${profit_loss:.2f} ({profit_pct:.2f}%)")
            print(f"Capital actual: ${self.capital:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error al procesar resultado de operación: {str(e)}")
            return {"success": False, "reason": f"Error: {str(e)}"}
    
    async def manage_funds(self) -> Dict[str, Any]:
        """
        Gestionar fondos según estrategia predefinida.
        
        Returns:
            Estado actualizado de fondos
        """
        # Si hay suficiente capital, retirar a reserva
        if self.capital > 200:
            profit = self.capital - 150  # Ganancia sobre el capital inicial
            
            # Determinar reserva objetivo según volatilidad
            avg_atr = np.mean(self.atr_history) if len(self.atr_history) > 1 else 0
            prev_avg_atr = np.mean(self.atr_history[:-1]) if len(self.atr_history) > 1 else 0
            reserve_target = 750 if avg_atr > 2 * prev_avg_atr else 500
            
            # Si la reserva no ha alcanzado el objetivo, retirar 50% de ganancias
            if self.reserve < reserve_target:
                withdrawal = profit * 0.5
                self.capital -= withdrawal
                self.reserve += withdrawal
                print(f"\n=== GESTIÓN DE FONDOS ===")
                print(f"Retiro a reserva: ${withdrawal:.2f}")
                print(f"Capital restante: ${self.capital:.2f}")
                print(f"Reserva actual: ${self.reserve:.2f} (Objetivo: ${reserve_target:.2f})")
            
            # Si se alcanzó el objetivo de reserva, completar ciclo
            if self.reserve >= reserve_target and not self.cycle_complete:
                self.cycle_complete = True
                
                # Distribuir fondos según estrategia
                self.emergency_fund += 20
                self.next_cycle += 20
                self.personal_use += 10
                
                print(f"\n=== ¡CICLO COMPLETADO! ===")
                print(f"Capital actual: ${self.capital:.2f}")
                print(f"Reserva: ${self.reserve:.2f}")
                print(f"Fondo de Emergencia: ${self.emergency_fund:.2f}")
                print(f"Próximo Ciclo: ${self.next_cycle:.2f}")
                print(f"Uso Personal: ${self.personal_use:.2f}")
        
        # Si el capital es bajo, usar fondos de próximo ciclo para recuperar
        elif self.capital < 120 and self.next_cycle >= 20:
            recovery_amount = min(self.next_cycle, 20)
            self.capital += recovery_amount
            self.next_cycle -= recovery_amount
            
            print(f"\n=== RECUPERACIÓN DE CAPITAL ===")
            print(f"Capital bajo detectado: ${self.capital - recovery_amount:.2f}")
            print(f"Transferido ${recovery_amount:.2f} desde Próximo Ciclo")
            print(f"Capital actualizado: ${self.capital:.2f}")
            print(f"Próximo Ciclo restante: ${self.next_cycle:.2f}")
        
        return {
            "capital": self.capital,
            "reserve": self.reserve,
            "emergency_fund": self.emergency_fund,
            "next_cycle": self.next_cycle,
            "personal_use": self.personal_use,
            "cycle_complete": self.cycle_complete
        }
    
    async def run_day_simulation(self, trades_per_day: int = 4):
        """
        Simular un día de trading.
        
        Args:
            trades_per_day: Número objetivo de operaciones por día
        """
        print(f"\n===== INICIO DEL DÍA DE TRADING =====")
        print(f"Capital inicial: ${self.capital:.2f}")
        print(f"Reserva: ${self.reserve:.2f}")
        print(f"Operaciones objetivo: {trades_per_day}")
        
        # Rotar símbolos y exchanges para distribuir operaciones
        symbols = self.get_symbols()
        executed_trades = 0
        
        for _ in range(trades_per_day * 2):  # Intentar el doble para lograr el objetivo
            if executed_trades >= trades_per_day:
                break
                
            # Seleccionar símbolo aleatorio
            symbol = random.choice(symbols)
            
            # Rotar exchange
            exchange = self.rotate_exchange()
            print(f"\nAnalizando {symbol} en {exchange}...")
            
            # Calcular trade
            trade = await self.calculate_trade(symbol)
            
            if trade["action"] == "buy":
                # Ejecutar operación
                result = await self.execute_trade(trade)
                if result.get("success") is not False:  # Evita contar cancelaciones
                    executed_trades += 1
            else:
                print(f"No hay señal de entrada para {symbol} en este momento")
            
            # Simular tiempo entre análisis
            print("Esperando próxima oportunidad...")
            await asyncio.sleep(1)  # En la simulación, esperar solo 1 segundo
        
        # Gestionar fondos al final del día
        await self.manage_funds()
        
        # Resumen del día
        print(f"\n===== RESUMEN DEL DÍA =====")
        print(f"Operaciones realizadas: {executed_trades}/{trades_per_day}")
        print(f"Capital final: ${self.capital:.2f}")
        print(f"Reserva: ${self.reserve:.2f}")
        print(f"Tasa de éxito: {self.metrics['success_rate']:.2f}%")
        print(f"Beneficio total: ${self.metrics['profit_total']:.2f}")
    
    async def run_cycle_simulation(self, days: int = 18, trades_per_day: int = 4):
        """
        Simular un ciclo completo de trading.
        
        Args:
            days: Número de días en el ciclo
            trades_per_day: Número objetivo de operaciones por día
        """
        print(f"\n======= INICIO DE CICLO DE TRADING ({days} días) =======")
        print(f"Capital inicial: ${self.capital:.2f}")
        print(f"Objetivo del ciclo: ${7500:.2f}")
        
        for day in range(1, days + 1):
            print(f"\n==== DÍA {day}/{days} ====")
            await self.run_day_simulation(trades_per_day)
            
            # Verificar si se alcanzó el objetivo del ciclo
            if self.cycle_complete:
                print(f"\n¡OBJETIVO ALCANZADO EN {day} DÍAS!")
                break
        
        # Resumen del ciclo
        print(f"\n======= RESUMEN DEL CICLO =======")
        print(f"Días transcurridos: {day}/{days}")
        print(f"Capital final: ${self.capital:.2f}")
        print(f"Reserva: ${self.reserve:.2f}")
        print(f"Operaciones totales: {self.metrics['trades_total']}")
        print(f"Tasa de éxito: {self.metrics['success_rate']:.2f}%")
        print(f"Beneficio promedio por operación: ${self.metrics['avg_profit_per_trade']:.2f}")
        print(f"Insights de Buddha utilizados: {self.metrics['buddha_insights_used']}")
        print(f"Ciclo completado: {'Sí' if self.cycle_complete else 'No'}")
        
        if not self.cycle_complete:
            print(f"\nEl objetivo (${7500:.2f}) no se alcanzó en el tiempo previsto.")
            print(f"Progreso: {(self.capital / 7500 * 100):.2f}% del objetivo")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas actuales de trading.
        
        Returns:
            Diccionario con métricas
        """
        return self.metrics
    
    def save_state(self, filename: str = "buddha_trader_state.json") -> bool:
        """
        Guardar estado actual para continuación posterior.
        
        Args:
            filename: Nombre del archivo para guardar estado
            
        Returns:
            True si se guardó correctamente
        """
        try:
            state = {
                "capital": self.capital,
                "reserve": self.reserve,
                "emergency_fund": self.emergency_fund,
                "next_cycle": self.next_cycle,
                "personal_use": self.personal_use,
                "metrics": self.metrics,
                "cycle_complete": self.cycle_complete,
                "timestamp": time.time()
            }
            
            with open(filename, "w") as f:
                json.dump(state, f, indent=4)
            
            logger.info(f"Estado guardado en {filename}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar estado: {str(e)}")
            return False
    
    def load_state(self, filename: str = "buddha_trader_state.json") -> bool:
        """
        Cargar estado previo.
        
        Args:
            filename: Nombre del archivo con estado guardado
            
        Returns:
            True si se cargó correctamente
        """
        try:
            with open(filename, "r") as f:
                state = json.load(f)
            
            self.capital = state["capital"]
            self.reserve = state["reserve"]
            self.emergency_fund = state["emergency_fund"]
            self.next_cycle = state["next_cycle"]
            self.personal_use = state["personal_use"]
            self.metrics = state["metrics"]
            self.cycle_complete = state["cycle_complete"]
            
            logger.info(f"Estado cargado desde {filename}")
            return True
        except Exception as e:
            logger.error(f"Error al cargar estado: {str(e)}")
            return False


# Función helper para ejecución simple
async def run_demo():
    """Ejecutar demo interactiva del Buddha Trader."""
    print("\n=== BUDDHA TRADER: DEMO INTERACTIVA ===")
    print("Esta demo simula la estrategia de trading potenciada por Buddha AI")
    
    # Solicitar parámetros iniciales
    capital = float(input("\nCapital inicial para trading (predeterminado: 150): ") or "150")
    days = int(input("Días de simulación (predeterminado: 5): ") or "5")
    trades_per_day = int(input("Trades por día (predeterminado: 4): ") or "4")
    
    # Inicializar trader
    trader = BuddhaTrader(capital=capital)
    
    # Verificar estado de Buddha
    print(f"\nBuddha AI {'habilitado' if trader.buddha_enabled else 'deshabilitado'}")
    
    # Ejecutar simulación
    await trader.run_cycle_simulation(days=days, trades_per_day=trades_per_day)
    
    # Guardar estado
    save = input("\n¿Guardar estado actual? (s/n): ").lower()
    if save == "s":
        filename = input("Nombre del archivo (predeterminado: buddha_trader_state.json): ") or "buddha_trader_state.json"
        trader.save_state(filename)
        print(f"Estado guardado en {filename}")


# Ejecución principal
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar demo
    asyncio.run(run_demo())