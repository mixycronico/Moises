"""
Analizador de datos on-chain para criptomonedas.

Este módulo proporciona funcionalidades para obtener y analizar
datos on-chain de blockchains a través de Glassnode y otras fuentes.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import urllib.parse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class OnchainAnalyzer:
    """
    Analizador de datos on-chain para criptomonedas.
    
    Proporciona herramientas para obtener y analizar métricas on-chain
    como la actividad de red, flujos de intercambios, y comportamiento de holders.
    """
    
    def __init__(self, 
                 glassnode_api_key: Optional[str] = None,
                 cache_dir: str = './cache/onchain',
                 use_mock: bool = False,
                 cache_expiry: int = 24*60*60):  # 24h por defecto
        """
        Inicializar analizador on-chain.
        
        Args:
            glassnode_api_key: API key para Glassnode
            cache_dir: Directorio para caché
            use_mock: Si es True, genera datos simulados cuando no hay acceso a API
            cache_expiry: Tiempo de expiración de caché en segundos
        """
        self.logger = logging.getLogger(__name__)
        self.glassnode_api_key = glassnode_api_key
        self.cache_dir = cache_dir
        self.use_mock = use_mock
        self.cache_expiry = cache_expiry
        
        # Crear directorio de caché si no existe
        os.makedirs(cache_dir, exist_ok=True)
        
        # Base URLs para APIs
        self.glassnode_base_url = "https://api.glassnode.com/v1/metrics"
        
        # Inicializar cliente HTTP asíncrono
        self.session = None
        
        self.logger.info("OnchainAnalyzer inicializado")
    
    async def _ensure_session(self) -> None:
        """Asegurar que existe una sesión HTTP."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self) -> None:
        """Cerrar la sesión HTTP."""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    async def get_glassnode_metric(self, 
                             endpoint: str, 
                             asset: str = 'BTC',
                             since: Optional[str] = None,
                             until: Optional[str] = None,
                             interval: str = '24h',
                             format: str = 'json') -> pd.DataFrame:
        """
        Obtener métrica de Glassnode.
        
        Args:
            endpoint: Endpoint de la métrica (ej. 'market/price_usd_close')
            asset: Activo (BTC, ETH, etc.)
            since: Fecha de inicio (ISO format)
            until: Fecha de fin (ISO format)
            interval: Intervalo (1h, 24h, etc.)
            format: Formato de respuesta (json, csv)
            
        Returns:
            DataFrame con la métrica
        """
        if self.glassnode_api_key is None:
            if self.use_mock:
                self.logger.warning("API key de Glassnode no configurada. Usando datos simulados.")
                return self._generate_mock_onchain_data(endpoint, asset, since, until, interval)
            else:
                raise ValueError("API key de Glassnode no configurada. Configure glassnode_api_key o establezca use_mock=True.")
        
        # Verificar si hay resultados en caché
        cache_key = f"{endpoint.replace('/', '_')}_{asset}_{interval}"
        if since:
            cache_key += f"_{since}"
        if until:
            cache_key += f"_{until}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # Usar caché si existe y no ha expirado
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < self.cache_expiry:
                self.logger.info(f"Usando datos on-chain en caché para {endpoint} ({asset})")
                df = pd.read_json(cache_file)
                return df
        
        # Asegurar que tenemos una sesión
        await self._ensure_session()
        
        # Construir URL
        url = f"{self.glassnode_base_url}/{endpoint}"
        
        # Construir parámetros
        params = {
            'a': asset,
            'api_key': self.glassnode_api_key,
            'i': interval,
            'f': format
        }
        
        if since:
            params['s'] = since
        
        if until:
            params['u'] = until
        
        # Realizar solicitud
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    if format == 'json':
                        data = await response.json()
                        df = pd.DataFrame(data)
                    elif format == 'csv':
                        text = await response.text()
                        df = pd.read_csv(StringIO(text))
                    else:
                        raise ValueError(f"Formato no soportado: {format}")
                    
                    # Guardar en caché
                    df.to_json(cache_file)
                    
                    return df
                else:
                    error_text = await response.text()
                    raise Exception(f"Error en API de Glassnode ({response.status}): {error_text}")
        except Exception as e:
            self.logger.error(f"Error obteniendo métrica {endpoint} para {asset}: {str(e)}")
            
            if self.use_mock:
                self.logger.warning("Usando datos simulados como respaldo.")
                return self._generate_mock_onchain_data(endpoint, asset, since, until, interval)
            else:
                raise
    
    def _generate_mock_onchain_data(self, 
                                  endpoint: str, 
                                  asset: str = 'BTC',
                                  since: Optional[str] = None,
                                  until: Optional[str] = None,
                                  interval: str = '24h') -> pd.DataFrame:
        """
        Generar datos on-chain simulados.
        
        Args:
            endpoint: Endpoint de la métrica
            asset: Activo
            since: Fecha de inicio
            until: Fecha de fin
            interval: Intervalo
            
        Returns:
            DataFrame con datos simulados
        """
        # Determinar fechas de inicio y fin
        if until:
            end_date = datetime.fromisoformat(until.replace('Z', '+00:00'))
        else:
            end_date = datetime.now()
        
        if since:
            start_date = datetime.fromisoformat(since.replace('Z', '+00:00'))
        else:
            # Por defecto, 90 días atrás
            start_date = end_date - timedelta(days=90)
        
        # Determinar delta de tiempo según intervalo
        if interval == '1h':
            delta = timedelta(hours=1)
        elif interval == '24h':
            delta = timedelta(days=1)
        elif interval == '1w':
            delta = timedelta(weeks=1)
        else:
            delta = timedelta(days=1)  # Por defecto diario
        
        # Generar timestamps
        timestamps = []
        current_date = start_date
        while current_date <= end_date:
            timestamps.append(current_date)
            current_date += delta
        
        # Preparar DataFrame
        df = pd.DataFrame()
        df['t'] = timestamps
        df['t'] = df['t'].apply(lambda x: int(x.timestamp()))
        
        # Generar valores según el tipo de métrica
        if 'price' in endpoint:
            # Precios con tendencia alcista y algo de volatilidad
            base_price = 30000 if asset == 'BTC' else (2000 if asset == 'ETH' else 100)
            trend = np.linspace(0, 0.3, len(timestamps))  # Tendencia alcista
            noise = np.random.normal(0, 0.02, len(timestamps))  # Volatilidad
            seasonal = 0.1 * np.sin(np.linspace(0, 4*np.pi, len(timestamps)))  # Patrón estacional
            
            # Combinar componentes
            values = base_price * (1 + trend + noise + seasonal)
            df['v'] = values
            
        elif 'volume' in endpoint:
            # Volumen con tendencia y picos aleatorios
            base_volume = 5000 if asset == 'BTC' else (2000 if asset == 'ETH' else 500)
            trend = np.linspace(0, 0.2, len(timestamps))
            noise = np.abs(np.random.normal(0, 0.3, len(timestamps)))  # Siempre positivo
            
            # Picos ocasionales (2x-3x)
            peaks = np.zeros(len(timestamps))
            peak_indices = np.random.choice(len(timestamps), size=int(len(timestamps)*0.05), replace=False)
            peaks[peak_indices] = np.random.uniform(1, 2, len(peak_indices))
            
            # Combinar componentes
            values = base_volume * (1 + trend + noise + peaks)
            df['v'] = values
            
        elif 'active_addresses' in endpoint or 'addresses' in endpoint:
            # Dirección activas con tendencia alcista
            base_value = 200000 if asset == 'BTC' else (100000 if asset == 'ETH' else 10000)
            trend = np.linspace(0, 0.5, len(timestamps))
            noise = np.random.normal(0, 0.05, len(timestamps))
            
            # Combinar componentes
            values = base_value * (1 + trend + noise)
            values = np.maximum(values, 0)  # Sin valores negativos
            df['v'] = values.astype(int)
            
        elif 'difficulty' in endpoint or 'hashrate' in endpoint:
            # Hashrate/dificultad con tendencia alcista y ajustes
            base_value = 1e8
            trend = np.linspace(0, 1.0, len(timestamps))
            
            # Ajustes de dificultad (cada ~2 semanas para BTC)
            steps = np.zeros(len(timestamps))
            step_indices = np.arange(0, len(timestamps), 14 // (1 if interval == '24h' else 14))
            for i in range(1, len(step_indices)):
                steps[step_indices[i]:] += np.random.uniform(-0.1, 0.3)
            
            # Combinar componentes
            values = base_value * (1 + trend + steps)
            df['v'] = values
            
        elif 'exchange' in endpoint or 'flow' in endpoint:
            # Flujos de exchange (positivos y negativos)
            base_value = 1000 if asset == 'BTC' else (5000 if asset == 'ETH' else 100)
            trend = np.zeros(len(timestamps))
            noise = np.random.normal(0, 1.0, len(timestamps))
            
            # Combinar componentes
            values = base_value * noise
            df['v'] = values
            
        elif 'sopr' in endpoint or 'profit' in endpoint:
            # Indicadores de rentabilidad
            base_value = 1.0
            noise = np.random.normal(0, 0.1, len(timestamps))
            cycle = 0.3 * np.sin(np.linspace(0, 3*np.pi, len(timestamps)))
            
            # Combinar componentes
            values = base_value + noise + cycle
            df['v'] = values
            
        else:
            # Valor genérico para otras métricas
            base_value = 1000
            trend = np.linspace(0, 0.2, len(timestamps))
            noise = np.random.normal(0, 0.1, len(timestamps))
            
            # Combinar componentes
            values = base_value * (1 + trend + noise)
            df['v'] = values
        
        return df
    
    async def get_network_activity(self, 
                             asset: str = 'BTC',
                             days_back: int = 90,
                             interval: str = '24h') -> Dict[str, pd.DataFrame]:
        """
        Obtener datos de actividad de red.
        
        Args:
            asset: Activo (BTC, ETH, etc.)
            days_back: Días hacia atrás
            interval: Intervalo de datos
            
        Returns:
            Diccionario con DataFrames de métricas
        """
        # Calcular fechas
        until = datetime.now().isoformat()
        since = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        # Métricas a obtener
        metrics = {
            'active_addresses': 'addresses/active_count',
            'new_addresses': 'addresses/new_non_zero_count',
            'transaction_count': 'transactions/count',
            'transaction_volume': 'transactions/transfers_volume_sum',
            'fees': 'fees/volume_sum'
        }
        
        # Obtener datos para cada métrica
        results = {}
        
        for name, endpoint in metrics.items():
            try:
                df = await self.get_glassnode_metric(
                    endpoint=endpoint,
                    asset=asset,
                    since=since,
                    until=until,
                    interval=interval
                )
                
                # Convertir timestamp a datetime
                df['t'] = pd.to_datetime(df['t'], unit='s')
                
                # Renombrar columnas
                df.rename(columns={'t': 'timestamp', 'v': name}, inplace=True)
                
                results[name] = df
                
            except Exception as e:
                self.logger.error(f"Error obteniendo métrica {name} para {asset}: {str(e)}")
        
        return results
    
    async def get_exchange_flows(self, 
                           asset: str = 'BTC',
                           days_back: int = 90,
                           interval: str = '24h') -> Dict[str, pd.DataFrame]:
        """
        Obtener datos de flujos de intercambios.
        
        Args:
            asset: Activo (BTC, ETH, etc.)
            days_back: Días hacia atrás
            interval: Intervalo de datos
            
        Returns:
            Diccionario con DataFrames de métricas
        """
        # Calcular fechas
        until = datetime.now().isoformat()
        since = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        # Métricas a obtener
        metrics = {
            'exchange_inflow': 'transactions/transfers_volume_to_exchanges_sum',
            'exchange_outflow': 'transactions/transfers_volume_from_exchanges_sum',
            'exchange_balance': 'distribution/balance_exchanges',
            'exchange_netflow': 'transactions/transfers_volume_exchanges_net'
        }
        
        # Obtener datos para cada métrica
        results = {}
        
        for name, endpoint in metrics.items():
            try:
                df = await self.get_glassnode_metric(
                    endpoint=endpoint,
                    asset=asset,
                    since=since,
                    until=until,
                    interval=interval
                )
                
                # Convertir timestamp a datetime
                df['t'] = pd.to_datetime(df['t'], unit='s')
                
                # Renombrar columnas
                df.rename(columns={'t': 'timestamp', 'v': name}, inplace=True)
                
                results[name] = df
                
            except Exception as e:
                self.logger.error(f"Error obteniendo métrica {name} para {asset}: {str(e)}")
        
        return results
    
    async def get_holder_behavior(self, 
                            asset: str = 'BTC',
                            days_back: int = 90,
                            interval: str = '24h') -> Dict[str, pd.DataFrame]:
        """
        Obtener datos de comportamiento de holders.
        
        Args:
            asset: Activo (BTC, ETH, etc.)
            days_back: Días hacia atrás
            interval: Intervalo de datos
            
        Returns:
            Diccionario con DataFrames de métricas
        """
        # Calcular fechas
        until = datetime.now().isoformat()
        since = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        # Métricas a obtener
        metrics = {
            'sopr': 'indicators/sopr',
            'supply_last_active_1y': 'supply/active_more_1y_percent',
            'profit_loss_ratio': 'indicators/realized_profit_loss_ratio',
            'long_term_holder_balance': 'supply/lth_sum',
            'short_term_holder_balance': 'supply/sth_sum'
        }
        
        # Obtener datos para cada métrica
        results = {}
        
        for name, endpoint in metrics.items():
            try:
                df = await self.get_glassnode_metric(
                    endpoint=endpoint,
                    asset=asset,
                    since=since,
                    until=until,
                    interval=interval
                )
                
                # Convertir timestamp a datetime
                df['t'] = pd.to_datetime(df['t'], unit='s')
                
                # Renombrar columnas
                df.rename(columns={'t': 'timestamp', 'v': name}, inplace=True)
                
                results[name] = df
                
            except Exception as e:
                self.logger.error(f"Error obteniendo métrica {name} para {asset}: {str(e)}")
        
        return results
    
    async def analyze_onchain_data(self, 
                             asset: str = 'BTC',
                             days_back: int = 90,
                             interval: str = '24h') -> Dict[str, Any]:
        """
        Analizar datos on-chain completos para un activo.
        
        Args:
            asset: Activo (BTC, ETH, etc.)
            days_back: Días hacia atrás
            interval: Intervalo de datos
            
        Returns:
            Diccionario con análisis completo
        """
        # Verificar si hay resultados en caché
        cache_file = os.path.join(self.cache_dir, f"analysis_{asset}_{days_back}d_{interval}.json")
        
        # Usar caché si existe y no ha expirado
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < self.cache_expiry:
                self.logger.info(f"Usando análisis on-chain en caché para {asset}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # Obtener datos de las diferentes categorías
        network_activity = await self.get_network_activity(asset, days_back, interval)
        exchange_flows = await self.get_exchange_flows(asset, days_back, interval)
        holder_behavior = await self.get_holder_behavior(asset, days_back, interval)
        
        # Verificar si tenemos suficientes datos
        if not network_activity or not exchange_flows or not holder_behavior:
            raise ValueError("No se pudieron obtener suficientes datos on-chain para el análisis")
        
        # Extraer métricas clave
        active_addresses = network_activity.get('active_addresses')
        transaction_volume = network_activity.get('transaction_volume')
        exchange_netflow = exchange_flows.get('exchange_netflow')
        sopr = holder_behavior.get('sopr')
        
        # Calcular señales de actividad de red
        network_activity_signal = 0
        if active_addresses is not None and transaction_volume is not None:
            # Calcular cambio relativo en direcciones activas (últimos 7 días vs 30 días)
            if len(active_addresses) >= 30:
                recent_activity = active_addresses['active_addresses'].iloc[-7:].mean()
                past_activity = active_addresses['active_addresses'].iloc[-30:-7].mean()
                
                if past_activity > 0:
                    activity_change = (recent_activity - past_activity) / past_activity
                    # Señal positiva si hay incremento en actividad
                    network_activity_signal = min(1, max(-1, activity_change * 5))  # Escalar entre -1 y 1
        
        # Calcular señales de flujos de exchange
        exchange_flow_signal = 0
        if exchange_netflow is not None:
            # Netflow acumulativo de las últimas 2 semanas
            if len(exchange_netflow) >= 14:
                recent_netflow = exchange_netflow['exchange_netflow'].iloc[-14:].sum()
                
                # Normalizar por rango típico
                typical_range = exchange_netflow['exchange_netflow'].abs().quantile(0.95)
                if typical_range > 0:
                    # Señal negativa si hay entradas netas (positivas) a exchanges
                    exchange_flow_signal = -min(1, max(-1, recent_netflow / typical_range))
        
        # Calcular señales de comportamiento de holders
        holder_behavior_signal = 0
        if sopr is not None:
            # SOPR > 1 indica ganancia, < 1 indica pérdida
            if len(sopr) >= 7:
                recent_sopr = sopr['sopr'].iloc[-7:].mean()
                
                # Calcular señal basada en SOPR
                if recent_sopr > 1.05:
                    # SOPR muy por encima de 1 puede indicar toma de ganancias (bajista)
                    holder_behavior_signal = -0.5
                elif 0.95 < recent_sopr < 1.05:
                    # SOPR cerca de 1 es neutral
                    holder_behavior_signal = 0
                else:
                    # SOPR por debajo de 1 indica capitulación/ventas con pérdida (potencialmente alcista)
                    holder_behavior_signal = 0.5
        
        # Calcular señal combinada
        combined_signal = (
            network_activity_signal * 0.3 +  # 30% de peso
            exchange_flow_signal * 0.4 +     # 40% de peso
            holder_behavior_signal * 0.3      # 30% de peso
        )
        
        # Determinar etiqueta de la señal
        if combined_signal > 0.2:
            signal_label = 'bullish'
        elif combined_signal < -0.2:
            signal_label = 'bearish'
        else:
            signal_label = 'neutral'
        
        # Preparar métricas clave para incluir en el resultado
        key_metrics = {}
        
        # Añadir métricas recientes de cada categoría
        for category, metrics in [
            ('network_activity', network_activity),
            ('exchange_flows', exchange_flows),
            ('holder_behavior', holder_behavior)
        ]:
            category_metrics = {}
            for name, df in metrics.items():
                if not df.empty:
                    # Obtener valor más reciente
                    latest = df.iloc[-1]
                    
                    # Calcular cambio respecto a 7 días atrás
                    if len(df) > 7:
                        previous = df.iloc[-8]
                        change_7d = (latest[name] - previous[name]) / previous[name] if previous[name] != 0 else 0
                    else:
                        change_7d = None
                    
                    # Calcular cambio respecto a 30 días atrás
                    if len(df) > 30:
                        previous = df.iloc[-31]
                        change_30d = (latest[name] - previous[name]) / previous[name] if previous[name] != 0 else 0
                    else:
                        change_30d = None
                    
                    # Guardar métrica
                    category_metrics[name] = {
                        'value': float(latest[name]),
                        'timestamp': latest['timestamp'].isoformat(),
                        'change_7d': float(change_7d) if change_7d is not None else None,
                        'change_30d': float(change_30d) if change_30d is not None else None
                    }
            
            key_metrics[category] = category_metrics
        
        # Preparar resultado
        result = {
            'asset': asset,
            'timestamp': datetime.now().isoformat(),
            'days_analyzed': days_back,
            'signals': {
                'network_activity': float(network_activity_signal),
                'exchange_flow': float(exchange_flow_signal),
                'holder_behavior': float(holder_behavior_signal),
                'combined': float(combined_signal),
                'label': signal_label
            },
            'key_metrics': key_metrics
        }
        
        # Guardar en caché
        with open(cache_file, 'w') as f:
            json.dump(result, f)
        
        return result
    
    def plot_onchain_signals(self, 
                            analysis_results: List[Dict[str, Any]],
                            price_data: Optional[pd.DataFrame] = None,
                            save_path: Optional[str] = None) -> str:
        """
        Generar gráfico de señales on-chain.
        
        Args:
            analysis_results: Lista de resultados de análisis on-chain
            price_data: DataFrame con datos de precio (opcional)
            save_path: Ruta donde guardar el gráfico
            
        Returns:
            Imagen en formato base64
        """
        if not analysis_results:
            return ""
        
        # Extraer datos
        timestamps = [datetime.fromisoformat(d['timestamp']) for d in analysis_results]
        combined_signals = [d['signals']['combined'] for d in analysis_results]
        network_signals = [d['signals']['network_activity'] for d in analysis_results]
        exchange_signals = [d['signals']['exchange_flow'] for d in analysis_results]
        holder_signals = [d['signals']['holder_behavior'] for d in analysis_results]
        
        # Crear figura
        if price_data is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = None
        
        # Graficar señales on-chain
        ax1.plot(timestamps, combined_signals, 'b-', linewidth=2, label='Señal combinada')
        ax1.plot(timestamps, network_signals, 'g--', alpha=0.7, label='Actividad de red')
        ax1.plot(timestamps, exchange_signals, 'r--', alpha=0.7, label='Flujos de exchanges')
        ax1.plot(timestamps, holder_signals, 'c--', alpha=0.7, label='Comportamiento de holders')
        
        # Añadir línea neutral
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Colorear áreas positivas y negativas para la señal combinada
        ax1.fill_between(timestamps, combined_signals, 0, where=np.array(combined_signals) > 0, color='green', alpha=0.2)
        ax1.fill_between(timestamps, combined_signals, 0, where=np.array(combined_signals) < 0, color='red', alpha=0.2)
        
        # Formato del gráfico de señales
        asset = analysis_results[0]['asset']
        ax1.set_title(f'Análisis On-Chain para {asset}')
        ax1.set_ylabel('Señal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graficar precio si está disponible
        if price_data is not None and ax2 is not None:
            # Filtrar datos de precio para el rango de fechas del análisis
            min_date = min(timestamps)
            max_date = max(timestamps)
            
            # Convertir índice a datetime si no lo es
            if not isinstance(price_data.index, pd.DatetimeIndex):
                if 'timestamp' in price_data.columns:
                    price_data.set_index('timestamp', inplace=True)
                else:
                    price_data.index = pd.to_datetime(price_data.index)
            
            # Filtrar
            filtered_price = price_data.loc[(price_data.index >= min_date) & (price_data.index <= max_date)]
            
            if not filtered_price.empty:
                # Graficar precio
                ax2.plot(filtered_price.index, filtered_price['close'], 'k-', label='Precio')
                
                # Formato del gráfico de precio
                ax2.set_ylabel('Precio')
                ax2.set_xlabel('Fecha')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Guardar gráfico si se solicita
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(img_str))
        
        return img_str
    
    def plot_network_activity(self, 
                            network_data: Dict[str, pd.DataFrame],
                            save_path: Optional[str] = None) -> str:
        """
        Generar gráfico de actividad de red.
        
        Args:
            network_data: Diccionario con DataFrames de actividad de red
            save_path: Ruta donde guardar el gráfico
            
        Returns:
            Imagen en formato base64
        """
        if not network_data:
            return ""
        
        # Determinar cuántos gráficos necesitamos
        n_metrics = len(network_data)
        if n_metrics == 0:
            return ""
        
        # Crear figura con subplots
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True)
        
        # Asegurar que axes sea una lista
        if n_metrics == 1:
            axes = [axes]
        
        # Graficar cada métrica
        for i, (name, df) in enumerate(network_data.items()):
            ax = axes[i]
            
            # Graficar métrica
            ax.plot(df['timestamp'], df[name], label=name.replace('_', ' ').title())
            
            # Formato del gráfico
            ax.set_title(name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Establecer formato de fecha solo en el último gráfico
            if i == n_metrics - 1:
                ax.set_xlabel('Fecha')
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Guardar gráfico si se solicita
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(img_str))
        
        return img_str