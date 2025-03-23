"""
Administrador de pipelines de trading para el sistema Genesis.

Este módulo orquesta el flujo completo de datos y decisiones de trading,
integrando todos los componentes: datos de mercado, análisis técnico,
análisis de sentimiento, análisis on-chain, ML/RL, y ejecución de órdenes.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

class PipelineManager:
    """
    Administrador de pipelines para el Sistema Genesis.
    
    Coordina el flujo de datos entre los diferentes componentes:
    1. Recolección de datos (mercado, sentimiento, on-chain)
    2. Procesamiento y análisis
    3. Generación de señales
    4. Ejecución de órdenes
    5. Análisis post-trade
    """
    
    def __init__(self):
        """Inicializar administrador de pipelines."""
        self.logger = logging.getLogger(__name__)
        
        # Componentes del pipeline
        self.components = {}
        
        # Estado del pipeline
        self.pipeline_running = False
        self.pipeline_thread = None
        
        # Lock para operaciones thread-safe
        self.lock = threading.RLock()
        
        # Configuración
        self.config = {
            'interval': 60,  # Intervalo en segundos
            'market_data_weight': 0.4,
            'sentiment_weight': 0.3,
            'onchain_weight': 0.3,
            'use_reinforcement_learning': True,
            'risk_management_enabled': True,
            'adaptive_scaling_enabled': True
        }
        
        # Estado de pipeline
        self.pipeline_state = {
            'last_run': None,
            'components_status': {},
            'signals': {},
            'errors': {}
        }
        
        self.logger.info("PipelineManager inicializado")
    
    def register_component(self, component_type: str, component: Any) -> bool:
        """
        Registrar un componente en el pipeline.
        
        Args:
            component_type: Tipo de componente
            component: Instancia del componente
            
        Returns:
            True si se registró correctamente
        """
        with self.lock:
            self.components[component_type] = component
            self.pipeline_state['components_status'][component_type] = 'registered'
            self.logger.info(f"Componente registrado: {component_type}")
            return True
    
    def unregister_component(self, component_type: str) -> bool:
        """
        Eliminar un componente del pipeline.
        
        Args:
            component_type: Tipo de componente
            
        Returns:
            True si se eliminó correctamente
        """
        with self.lock:
            if component_type in self.components:
                del self.components[component_type]
                self.pipeline_state['components_status'][component_type] = 'unregistered'
                self.logger.info(f"Componente eliminado: {component_type}")
                return True
            return False
    
    def get_component(self, component_type: str) -> Optional[Any]:
        """
        Obtener un componente registrado.
        
        Args:
            component_type: Tipo de componente
            
        Returns:
            Instancia del componente o None si no existe
        """
        with self.lock:
            return self.components.get(component_type)
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Actualizar configuración del pipeline.
        
        Args:
            config_updates: Diccionario con actualizaciones
            
        Returns:
            True si se actualizó correctamente
        """
        with self.lock:
            for key, value in config_updates.items():
                if key in self.config:
                    self.config[key] = value
                    self.logger.info(f"Configuración actualizada: {key}={value}")
                else:
                    self.logger.warning(f"Configuración desconocida: {key}")
            
            # Normalizar pesos si se actualizaron
            self._normalize_weights()
            
            return True
    
    def _normalize_weights(self) -> None:
        """Normalizar pesos de análisis para que sumen 1.0."""
        weight_keys = ['market_data_weight', 'sentiment_weight', 'onchain_weight']
        
        # Calcular suma actual
        total_weight = sum(self.config[k] for k in weight_keys)
        
        # Normalizar si es necesario
        if abs(total_weight - 1.0) > 1e-6:
            for key in weight_keys:
                self.config[key] /= total_weight
    
    def start_pipeline(self) -> bool:
        """
        Iniciar el pipeline de procesamiento.
        
        Returns:
            True si se inició correctamente
        """
        with self.lock:
            if self.pipeline_running:
                self.logger.warning("Pipeline ya está en ejecución")
                return False
            
            # Verificar componentes requeridos
            required_components = ['market_data', 'signals', 'orders']
            for comp in required_components:
                if comp not in self.components:
                    self.logger.error(f"Componente requerido no registrado: {comp}")
                    return False
            
            # Iniciar pipeline en thread separado
            self.pipeline_running = True
            self.pipeline_thread = threading.Thread(target=self._pipeline_loop)
            self.pipeline_thread.daemon = True
            self.pipeline_thread.start()
            
            self.logger.info("Pipeline iniciado")
            return True
    
    def stop_pipeline(self) -> bool:
        """
        Detener el pipeline de procesamiento.
        
        Returns:
            True si se detuvo correctamente
        """
        with self.lock:
            if not self.pipeline_running:
                self.logger.warning("Pipeline no está en ejecución")
                return False
            
            # Detener pipeline
            self.pipeline_running = False
            
            # Esperar a que el thread termine
            if self.pipeline_thread and self.pipeline_thread.is_alive():
                self.pipeline_thread.join(timeout=5.0)
            
            self.logger.info("Pipeline detenido")
            return True
    
    def _pipeline_loop(self) -> None:
        """Loop principal del pipeline."""
        self.logger.info("Pipeline loop iniciado")
        
        while self.pipeline_running:
            try:
                # Ejecutar un ciclo del pipeline
                asyncio.run(self._execute_pipeline_cycle())
                
                # Actualizar estado
                with self.lock:
                    self.pipeline_state['last_run'] = datetime.now().isoformat()
                
            except Exception as e:
                self.logger.error(f"Error en ciclo de pipeline: {str(e)}")
                with self.lock:
                    self.pipeline_state['errors']['pipeline_cycle'] = str(e)
            
            # Esperar hasta el próximo ciclo
            time.sleep(self.config['interval'])
        
        self.logger.info("Pipeline loop terminado")
    
    async def _execute_pipeline_cycle(self) -> None:
        """Ejecutar un ciclo completo del pipeline."""
        # 1. Obtener datos de mercado
        market_data = await self._fetch_market_data()
        
        # 2. Obtener análisis de sentimiento si está disponible
        sentiment_data = await self._fetch_sentiment_data()
        
        # 3. Obtener análisis on-chain si está disponible
        onchain_data = await self._fetch_onchain_data()
        
        # 4. Procesar datos y generar señales
        signals = await self._generate_signals(market_data, sentiment_data, onchain_data)
        
        # 5. Aplicar gestión de riesgos
        risk_adjusted_signals = await self._apply_risk_management(signals)
        
        # 6. Ejecutar órdenes basadas en señales
        execution_results = await self._execute_orders(risk_adjusted_signals)
        
        # 7. Actualizar estado del portfolio
        await self._update_portfolio_state(execution_results)
        
        # 8. Aplicar escalado adaptativo si está habilitado
        if self.config['adaptive_scaling_enabled']:
            await self._apply_adaptive_scaling()
    
    async def _fetch_market_data(self) -> Dict[str, Any]:
        """
        Obtener datos de mercado de los diferentes exchanges.
        
        Returns:
            Diccionario con datos de mercado
        """
        market_data_component = self.get_component('market_data')
        if market_data_component is None:
            self.logger.error("Componente market_data no registrado")
            raise ValueError("Componente market_data no disponible")
        
        try:
            # Obtener datos de mercado
            market_data = await market_data_component.get_market_data()
            
            with self.lock:
                self.pipeline_state['components_status']['market_data'] = 'ok'
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de mercado: {str(e)}")
            
            with self.lock:
                self.pipeline_state['components_status']['market_data'] = 'error'
                self.pipeline_state['errors']['market_data'] = str(e)
            
            # Devolver datos vacíos
            return {}
    
    async def _fetch_sentiment_data(self) -> Optional[Dict[str, Any]]:
        """
        Obtener análisis de sentimiento.
        
        Returns:
            Diccionario con datos de sentimiento o None si no está disponible
        """
        sentiment_component = self.get_component('sentiment')
        if sentiment_component is None:
            self.logger.info("Componente sentiment no registrado")
            return None
        
        try:
            # Obtener análisis de sentimiento
            sentiment_data = await sentiment_component.get_sentiment_data()
            
            with self.lock:
                self.pipeline_state['components_status']['sentiment'] = 'ok'
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de sentimiento: {str(e)}")
            
            with self.lock:
                self.pipeline_state['components_status']['sentiment'] = 'error'
                self.pipeline_state['errors']['sentiment'] = str(e)
            
            # Devolver None
            return None
    
    async def _fetch_onchain_data(self) -> Optional[Dict[str, Any]]:
        """
        Obtener análisis on-chain.
        
        Returns:
            Diccionario con datos on-chain o None si no está disponible
        """
        onchain_component = self.get_component('onchain')
        if onchain_component is None:
            self.logger.info("Componente onchain no registrado")
            return None
        
        try:
            # Obtener análisis on-chain
            onchain_data = await onchain_component.get_onchain_data()
            
            with self.lock:
                self.pipeline_state['components_status']['onchain'] = 'ok'
            
            return onchain_data
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos on-chain: {str(e)}")
            
            with self.lock:
                self.pipeline_state['components_status']['onchain'] = 'error'
                self.pipeline_state['errors']['onchain'] = str(e)
            
            # Devolver None
            return None
    
    async def _generate_signals(self, 
                              market_data: Dict[str, Any],
                              sentiment_data: Optional[Dict[str, Any]] = None,
                              onchain_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generar señales de trading combinando diferentes fuentes.
        
        Args:
            market_data: Datos de mercado
            sentiment_data: Datos de sentimiento (opcional)
            onchain_data: Datos on-chain (opcional)
            
        Returns:
            Diccionario con señales de trading
        """
        signals_component = self.get_component('signals')
        if signals_component is None:
            self.logger.error("Componente signals no registrado")
            raise ValueError("Componente signals no disponible")
        
        # Preparar datos para generar señales
        input_data = {
            'market_data': market_data,
            'sentiment_data': sentiment_data,
            'onchain_data': onchain_data,
            'weights': {
                'market_data': self.config['market_data_weight'],
                'sentiment': self.config['sentiment_weight'],
                'onchain': self.config['onchain_weight']
            }
        }
        
        try:
            # Añadir datos de RL si está habilitado
            if self.config['use_reinforcement_learning']:
                rl_component = self.get_component('reinforcement_learning')
                if rl_component is not None:
                    rl_signals = await rl_component.get_signals(market_data)
                    input_data['rl_signals'] = rl_signals
            
            # Generar señales
            signals = await signals_component.generate_signals(input_data)
            
            with self.lock:
                self.pipeline_state['components_status']['signals'] = 'ok'
                self.pipeline_state['signals'] = signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generando señales: {str(e)}")
            
            with self.lock:
                self.pipeline_state['components_status']['signals'] = 'error'
                self.pipeline_state['errors']['signals'] = str(e)
            
            # Devolver señales vacías
            return {}
    
    async def _apply_risk_management(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplicar gestión de riesgos a las señales.
        
        Args:
            signals: Señales de trading
            
        Returns:
            Señales ajustadas según riesgo
        """
        if not self.config['risk_management_enabled']:
            return signals
        
        risk_component = self.get_component('risk_management')
        if risk_component is None:
            self.logger.info("Componente risk_management no registrado")
            return signals
        
        try:
            # Aplicar gestión de riesgos
            adjusted_signals = await risk_component.adjust_signals(signals)
            
            with self.lock:
                self.pipeline_state['components_status']['risk_management'] = 'ok'
            
            return adjusted_signals
            
        except Exception as e:
            self.logger.error(f"Error aplicando gestión de riesgos: {str(e)}")
            
            with self.lock:
                self.pipeline_state['components_status']['risk_management'] = 'error'
                self.pipeline_state['errors']['risk_management'] = str(e)
            
            # Devolver señales originales
            return signals
    
    async def _execute_orders(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar órdenes basadas en señales.
        
        Args:
            signals: Señales de trading
            
        Returns:
            Resultados de ejecución
        """
        orders_component = self.get_component('orders')
        if orders_component is None:
            self.logger.error("Componente orders no registrado")
            raise ValueError("Componente orders no disponible")
        
        try:
            # Ejecutar órdenes
            execution_results = await orders_component.execute_orders(signals)
            
            with self.lock:
                self.pipeline_state['components_status']['orders'] = 'ok'
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Error ejecutando órdenes: {str(e)}")
            
            with self.lock:
                self.pipeline_state['components_status']['orders'] = 'error'
                self.pipeline_state['errors']['orders'] = str(e)
            
            # Devolver resultados vacíos
            return {}
    
    async def _update_portfolio_state(self, execution_results: Dict[str, Any]) -> None:
        """
        Actualizar estado del portfolio después de ejecutar órdenes.
        
        Args:
            execution_results: Resultados de ejecución
        """
        portfolio_component = self.get_component('portfolio')
        if portfolio_component is None:
            self.logger.info("Componente portfolio no registrado")
            return
        
        try:
            # Actualizar portfolio
            await portfolio_component.update_state(execution_results)
            
            with self.lock:
                self.pipeline_state['components_status']['portfolio'] = 'ok'
                
        except Exception as e:
            self.logger.error(f"Error actualizando portfolio: {str(e)}")
            
            with self.lock:
                self.pipeline_state['components_status']['portfolio'] = 'error'
                self.pipeline_state['errors']['portfolio'] = str(e)
    
    async def _apply_adaptive_scaling(self) -> None:
        """Aplicar escalado adaptativo al sistema."""
        scaling_component = self.get_component('adaptive_scaling')
        if scaling_component is None:
            self.logger.info("Componente adaptive_scaling no registrado")
            return
        
        try:
            # Obtener estado del portfolio
            portfolio_component = self.get_component('portfolio')
            if portfolio_component is None:
                return
            
            portfolio_state = await portfolio_component.get_state()
            
            # Aplicar escalado adaptativo
            scaling_result = await scaling_component.adjust_allocation(portfolio_state)
            
            with self.lock:
                self.pipeline_state['components_status']['adaptive_scaling'] = 'ok'
                
        except Exception as e:
            self.logger.error(f"Error aplicando escalado adaptativo: {str(e)}")
            
            with self.lock:
                self.pipeline_state['components_status']['adaptive_scaling'] = 'error'
                self.pipeline_state['errors']['adaptive_scaling'] = str(e)
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """
        Obtener estado actual del pipeline.
        
        Returns:
            Diccionario con estado del pipeline
        """
        with self.lock:
            # Copia del estado para evitar modificaciones externas
            return json.loads(json.dumps(self.pipeline_state))
    
    def run_manual_cycle(self) -> Dict[str, Any]:
        """
        Ejecutar un ciclo de pipeline manualmente.
        
        Returns:
            Estado resultante del pipeline
        """
        try:
            # Ejecutar ciclo
            asyncio.run(self._execute_pipeline_cycle())
            
            # Devolver estado actualizado
            return self.get_pipeline_state()
            
        except Exception as e:
            self.logger.error(f"Error en ciclo manual: {str(e)}")
            
            with self.lock:
                self.pipeline_state['errors']['manual_cycle'] = str(e)
            
            return self.get_pipeline_state()
    
    async def integrate_sentiment_onchain(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Integrar análisis de sentimiento y on-chain para símbolos específicos.
        
        Args:
            symbols: Lista de símbolos a analizar
            
        Returns:
            Diccionario con análisis integrado por símbolo
        """
        # Verificar componentes
        sentiment_component = self.get_component('sentiment')
        onchain_component = self.get_component('onchain')
        
        if sentiment_component is None and onchain_component is None:
            self.logger.warning("Componentes sentiment y onchain no disponibles")
            return {}
        
        # Resultados por símbolo
        results = {}
        
        # Procesar cada símbolo
        for symbol in symbols:
            symbol_result = {'symbol': symbol}
            
            # Obtener análisis de sentimiento
            if sentiment_component is not None:
                try:
                    sentiment_data = await sentiment_component.analyze_social_sentiment(symbol)
                    symbol_result['sentiment'] = {
                        'score': sentiment_data.get('weighted_sentiment_score', 0),
                        'label': sentiment_data.get('sentiment_label', 'neutral'),
                        'distribution': sentiment_data.get('sentiment_distribution', {})
                    }
                except Exception as e:
                    self.logger.error(f"Error en análisis de sentimiento para {symbol}: {str(e)}")
                    symbol_result['sentiment'] = {
                        'score': 0,
                        'label': 'neutral',
                        'error': str(e)
                    }
            
            # Obtener análisis on-chain
            if onchain_component is not None:
                try:
                    onchain_data = await onchain_component.analyze_onchain_data(symbol)
                    symbol_result['onchain'] = {
                        'combined_signal': onchain_data.get('signals', {}).get('combined', 0),
                        'label': onchain_data.get('signals', {}).get('label', 'neutral'),
                        'key_metrics': onchain_data.get('key_metrics', {})
                    }
                except Exception as e:
                    self.logger.error(f"Error en análisis on-chain para {symbol}: {str(e)}")
                    symbol_result['onchain'] = {
                        'combined_signal': 0,
                        'label': 'neutral',
                        'error': str(e)
                    }
            
            # Calcular señal combinada de sentimiento y on-chain
            sentiment_score = symbol_result.get('sentiment', {}).get('score', 0)
            onchain_signal = symbol_result.get('onchain', {}).get('combined_signal', 0)
            
            # Ponderación
            sentiment_weight = self.config['sentiment_weight'] / (self.config['sentiment_weight'] + self.config['onchain_weight'])
            onchain_weight = 1.0 - sentiment_weight
            
            # Señal combinada normalizada entre -1 y 1
            combined_signal = sentiment_score * sentiment_weight + onchain_signal * onchain_weight
            
            # Determinar etiqueta
            if combined_signal > 0.2:
                combined_label = 'bullish'
            elif combined_signal < -0.2:
                combined_label = 'bearish'
            else:
                combined_label = 'neutral'
            
            # Añadir resultado combinado
            symbol_result['combined'] = {
                'signal': combined_signal,
                'label': combined_label
            }
            
            # Añadir al resultado
            results[symbol] = symbol_result
        
        return results
    
    def get_component_status(self, component_type: str) -> Dict[str, Any]:
        """
        Obtener estado de un componente específico.
        
        Args:
            component_type: Tipo de componente
            
        Returns:
            Diccionario con estado del componente
        """
        component = self.get_component(component_type)
        if component is None:
            return {'status': 'not_registered'}
        
        with self.lock:
            status = self.pipeline_state['components_status'].get(component_type, 'unknown')
            error = self.pipeline_state['errors'].get(component_type)
        
        result = {'status': status}
        
        if error:
            result['error'] = error
        
        # Obtener información adicional del componente si tiene método get_status
        if hasattr(component, 'get_status') and callable(component.get_status):
            try:
                component_status = component.get_status()
                result['details'] = component_status
            except Exception as e:
                self.logger.error(f"Error obteniendo estado del componente {component_type}: {str(e)}")
                result['details_error'] = str(e)
        
        return result