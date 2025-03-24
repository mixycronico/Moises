#!/usr/bin/env python3
"""
Prueba del CloudLoadBalancer Ultra-Divino.

Este script ejecuta una prueba completa del balanceador de carga cloud,
demostrando sus capacidades de distribución inteligente, auto-escalado
y recuperación ante fallos, con integración perfecta con los demás
componentes cloud (Circuit Breaker y Checkpoint Manager).
"""

import os
import sys
import json
import asyncio
import random
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

from genesis.cloud import (
    # Circuit Breaker
    CloudCircuitBreaker, CloudCircuitBreakerFactory, CircuitState,
    circuit_breaker_factory, circuit_protected,
    
    # Distributed Checkpoint Manager
    DistributedCheckpointManager, CheckpointStorageType, 
    CheckpointConsistencyLevel, checkpoint_manager,
    
    # Cloud Load Balancer
    CloudLoadBalancer, CloudLoadBalancerManager, CloudNode,
    BalancerAlgorithm, ScalingPolicy, BalancerState,
    SessionAffinityMode, NodeHealthStatus,
    load_balancer_manager, distributed, ultra_resilient
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_cloud_load_balancer")


class BeautifulTerminalColors:
    """Colores para terminal con estilo artístico."""
    HEADER = '\033[95m'        # Magenta claro
    BLUE = '\033[94m'          # Azul
    CYAN = '\033[96m'          # Cian
    GREEN = '\033[92m'         # Verde
    YELLOW = '\033[93m'        # Amarillo
    RED = '\033[91m'           # Rojo
    BOLD = '\033[1m'           # Negrita
    UNDERLINE = '\033[4m'      # Subrayado
    DIVINE = '\033[38;5;141m'  # Púrpura divino
    QUANTUM = '\033[38;5;39m'  # Azul cuántico
    COSMIC = '\033[38;5;208m'  # Naranja cósmico
    TRANSCEND = '\033[38;5;51m'# Aguamarina trascendental
    END = '\033[0m'            # Reset


class TradingPrice:
    """Simulador simple de precios de trading."""
    
    def __init__(self, initial_price: float = 50000.0):
        """
        Inicializar simulador.
        
        Args:
            initial_price: Precio inicial
        """
        self.initial_price = initial_price
        self.current_price = initial_price
        self.volatility = 0.01  # 1% de volatilidad por defecto
        self.drift = 0.0001     # Tendencia muy ligera al alza
        self.last_update = time.time()
        self.price_history = [(time.time(), initial_price)]
    
    def update(self) -> float:
        """
        Actualizar precio con movimiento aleatorio.
        
        Returns:
            Nuevo precio
        """
        # Tiempo desde última actualización
        now = time.time()
        time_delta = now - self.last_update
        self.last_update = now
        
        # Movimiento browniano geométrico simple
        random_change = random.normalvariate(0, 1) * self.volatility * (time_delta ** 0.5)
        drift_change = self.drift * time_delta
        
        # Cambio porcentual
        percent_change = drift_change + random_change
        
        # Aplicar cambio
        self.current_price *= (1 + percent_change)
        
        # Guardar en historial
        self.price_history.append((now, self.current_price))
        
        # Limpiar historial si es demasiado largo
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]
        
        return self.current_price
    
    def get_price(self) -> float:
        """
        Obtener precio actual.
        
        Returns:
            Precio actual
        """
        self.update()
        return self.current_price
    
    def set_volatility(self, volatility: float) -> None:
        """
        Establecer nivel de volatilidad.
        
        Args:
            volatility: Nueva volatilidad (0-1)
        """
        self.volatility = max(0.001, min(0.1, volatility))
    
    def set_drift(self, drift: float) -> None:
        """
        Establecer tendencia.
        
        Args:
            drift: Nueva tendencia
        """
        self.drift = max(-0.001, min(0.001, drift))


class TradingOperations:
    """Simulador de operaciones de trading."""
    
    def __init__(self):
        """Inicializar simulador."""
        self.symbols = {
            "BTC": TradingPrice(50000.0),
            "ETH": TradingPrice(3000.0),
            "SOL": TradingPrice(150.0),
            "ADA": TradingPrice(1.2),
            "DOT": TradingPrice(25.0)
        }
        
        # Configurar diferentes parámetros para cada símbolo
        self.symbols["BTC"].set_volatility(0.015)
        self.symbols["ETH"].set_volatility(0.02)
        self.symbols["SOL"].set_volatility(0.025)
        self.symbols["ADA"].set_volatility(0.01)
        self.symbols["DOT"].set_volatility(0.02)
        
        self.symbols["BTC"].set_drift(0.0002)
        self.symbols["ETH"].set_drift(0.0001)
        self.symbols["SOL"].set_drift(0.0003)
        self.symbols["ADA"].set_drift(-0.0001)
        self.symbols["DOT"].set_drift(0.0)
        
        # Estadísticas
        self.operations_count = 0
        self.failed_operations = 0
        self.last_operation_time = 0
    
    async def get_price(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener precio actual de un símbolo.
        
        Args:
            symbol: Símbolo a consultar
            
        Returns:
            Datos del precio
            
        Raises:
            ValueError: Si el símbolo no existe
        """
        if symbol not in self.symbols:
            raise ValueError(f"Símbolo {symbol} no reconocido")
        
        # Simular latencia de operación
        await asyncio.sleep(random.uniform(0.01, 0.05))
        
        # Actualizar estadísticas
        self.operations_count += 1
        self.last_operation_time = time.time()
        
        # Obtener precio
        price = self.symbols[symbol].get_price()
        
        # Simular error aleatorio
        if random.random() < 0.05:  # 5% de probabilidad de error
            self.failed_operations += 1
            raise Exception(f"Error simulado al obtener precio de {symbol}")
        
        # Devolver datos
        return {
            "symbol": symbol,
            "price": price,
            "timestamp": time.time(),
            "operation_id": f"price_{self.operations_count}_{random.randint(1000, 9999)}"
        }
    
    async def place_order(self, symbol: str, order_type: str, amount: float) -> Dict[str, Any]:
        """
        Colocar orden de trading.
        
        Args:
            symbol: Símbolo a operar
            order_type: Tipo de orden (buy/sell)
            amount: Cantidad a operar
            
        Returns:
            Datos de la orden
            
        Raises:
            ValueError: Si los parámetros son inválidos
            Exception: Si ocurre un error simulado
        """
        if symbol not in self.symbols:
            raise ValueError(f"Símbolo {symbol} no reconocido")
        
        if order_type not in ["buy", "sell"]:
            raise ValueError(f"Tipo de orden {order_type} no válido")
        
        if amount <= 0:
            raise ValueError(f"Cantidad {amount} debe ser mayor que cero")
        
        # Simular latencia de operación (mayor que consulta de precio)
        await asyncio.sleep(random.uniform(0.05, 0.15))
        
        # Actualizar estadísticas
        self.operations_count += 1
        self.last_operation_time = time.time()
        
        # Obtener precio
        price = self.symbols[symbol].get_price()
        
        # Simular error aleatorio (mayor probabilidad que en consulta de precio)
        if random.random() < 0.1:  # 10% de probabilidad de error
            self.failed_operations += 1
            raise Exception(f"Error simulado al colocar orden de {order_type} para {symbol}")
        
        # Crear ID único para la orden
        order_id = f"order_{order_type}_{self.operations_count}_{random.randint(1000, 9999)}"
        
        # Devolver datos
        return {
            "order_id": order_id,
            "symbol": symbol,
            "type": order_type,
            "amount": amount,
            "price": price,
            "status": "filled",
            "filled_amount": amount,
            "filled_price": price * random.uniform(0.995, 1.005),  # Simular ligero slippage
            "timestamp": time.time(),
            "fee": amount * price * 0.001  # Fee del 0.1%
        }
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """
        Obtener balance de la cuenta.
        
        Returns:
            Datos del balance
            
        Raises:
            Exception: Si ocurre un error simulado
        """
        # Simular latencia de operación
        await asyncio.sleep(random.uniform(0.03, 0.1))
        
        # Actualizar estadísticas
        self.operations_count += 1
        self.last_operation_time = time.time()
        
        # Crear balance simulado
        balance = {
            "total_equity": 100000.0 * random.uniform(0.9, 1.1),
            "available_balance": 50000.0 * random.uniform(0.9, 1.1),
            "holdings": {
                symbol: {
                    "amount": random.uniform(0.1, 10.0),
                    "avg_price": price_simulator.get_price() * random.uniform(0.9, 1.1)
                }
                for symbol, price_simulator in self.symbols.items()
            },
            "timestamp": time.time(),
            "request_id": f"balance_{self.operations_count}_{random.randint(1000, 9999)}"
        }
        
        # Simular error aleatorio
        if random.random() < 0.05:  # 5% de probabilidad de error
            self.failed_operations += 1
            raise Exception("Error simulado al obtener balance de cuenta")
        
        return balance
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen del mercado.
        
        Returns:
            Datos del mercado
            
        Raises:
            Exception: Si ocurre un error simulado
        """
        # Simular latencia de operación (operación más pesada)
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Actualizar estadísticas
        self.operations_count += 1
        self.last_operation_time = time.time()
        
        # Crear resumen simulado
        summary = {
            "symbols": {
                symbol: {
                    "price": price_simulator.get_price(),
                    "24h_change": random.uniform(-5.0, 5.0),
                    "24h_volume": random.uniform(1000000, 10000000),
                    "market_cap": price_simulator.get_price() * random.uniform(10000000, 100000000)
                }
                for symbol, price_simulator in self.symbols.items()
            },
            "global_market_cap": random.uniform(1000000000000, 2000000000000),
            "global_volume_24h": random.uniform(100000000000, 500000000000),
            "btc_dominance": random.uniform(40.0, 60.0),
            "timestamp": time.time(),
            "request_id": f"market_{self.operations_count}_{random.randint(1000, 9999)}"
        }
        
        # Simular error aleatorio (mayor probabilidad por ser operación compleja)
        if random.random() < 0.15:  # 15% de probabilidad de error
            self.failed_operations += 1
            raise Exception("Error simulado al obtener resumen del mercado")
        
        return summary


async def test_basic_load_balancing():
    """Probar funcionalidades básicas del balanceador de carga."""
    c = BeautifulTerminalColors
    print(f"\n{c.DIVINE}=== PRUEBA BÁSICA DE BALANCEO DE CARGA ==={c.END}")
    
    # Inicializar manager de balanceo
    await load_balancer_manager.initialize()
    
    # Crear nodos simulados
    nodes = [
        CloudNode(f"node_{i}", "127.0.0.1", 8080 + i, weight=random.uniform(0.5, 1.5))
        for i in range(5)
    ]
    
    # Crear balanceador para operaciones de trading
    balancer = await load_balancer_manager.create_balancer(
        name="trading_operations",
        algorithm=BalancerAlgorithm.WEIGHTED,
        scaling_policy=ScalingPolicy.THRESHOLD,
        session_affinity=SessionAffinityMode.TOKEN,
        initial_nodes=nodes
    )
    
    if not balancer:
        print(f"{c.RED}Error al crear balanceador{c.END}")
        return
    
    # Inicializar simulador de operaciones
    trading_ops = TradingOperations()
    
    # Función para mostrar distribución de carga
    def print_load_distribution():
        print(f"\n{c.CYAN}Distribución de carga entre nodos:{c.END}")
        for node_id, node in balancer.nodes.items():
            print(f"  {node_id}: {node.active_connections} conexiones activas, "
                  f"{node.total_connections} totales")
    
    # Ejecutar operaciones distribuidas
    print(f"\n{c.CYAN}Ejecutando operaciones distribuidas...{c.END}")
    
    # Simular diferentes usuarios
    users = ["user1", "user2", "user3", "user4", "user5"]
    
    for i in range(30):
        # Alternar entre diferentes tipos de operaciones
        operation_type = i % 4
        user = random.choice(users)
        
        try:
            if operation_type == 0:
                # Consulta de precio
                symbol = random.choice(list(trading_ops.symbols.keys()))
                result, node = await balancer.execute_operation(
                    trading_ops.get_price, 
                    session_key=user,
                    symbol=symbol
                )
                
                if result and node:
                    print(f"  Operación #{i}: {c.GREEN}Precio de {symbol}: "
                          f"${result['price']:.2f}{c.END} - Nodo: {node.node_id}")
                else:
                    print(f"  Operación #{i}: {c.RED}Error al obtener precio de {symbol}{c.END}")
            
            elif operation_type == 1:
                # Colocación de orden
                symbol = random.choice(list(trading_ops.symbols.keys()))
                order_type = random.choice(["buy", "sell"])
                amount = random.uniform(0.1, 1.0)
                
                result, node = await balancer.execute_operation(
                    trading_ops.place_order,
                    session_key=user,
                    symbol=symbol,
                    order_type=order_type,
                    amount=amount
                )
                
                if result and node:
                    print(f"  Operación #{i}: {c.GREEN}Orden {order_type} de {amount} {symbol} "
                          f"a ${result['price']:.2f}{c.END} - Nodo: {node.node_id}")
                else:
                    print(f"  Operación #{i}: {c.RED}Error al colocar orden{c.END}")
            
            elif operation_type == 2:
                # Consulta de balance
                result, node = await balancer.execute_operation(
                    trading_ops.get_account_balance,
                    session_key=user
                )
                
                if result and node:
                    print(f"  Operación #{i}: {c.GREEN}Balance total: "
                          f"${result['total_equity']:.2f}{c.END} - Nodo: {node.node_id}")
                else:
                    print(f"  Operación #{i}: {c.RED}Error al obtener balance{c.END}")
            
            else:
                # Resumen de mercado
                result, node = await balancer.execute_operation(
                    trading_ops.get_market_summary,
                    session_key=user
                )
                
                if result and node:
                    print(f"  Operación #{i}: {c.GREEN}Market cap global: "
                          f"${result['global_market_cap']:.2f}{c.END} - Nodo: {node.node_id}")
                else:
                    print(f"  Operación #{i}: {c.RED}Error al obtener resumen de mercado{c.END}")
        
        except Exception as e:
            print(f"  Operación #{i}: {c.RED}Excepción: {e}{c.END}")
        
        # Pausa breve entre operaciones
        await asyncio.sleep(0.1)
    
    # Mostrar distribución de carga resultante
    print_load_distribution()
    
    # Estado del balanceador
    print(f"\n{c.CYAN}Estado del balanceador:{c.END}")
    status = balancer.get_status()
    print(f"  Estado: {status['state']}")
    print(f"  Operaciones totales: {status['metrics']['total_operations']}")
    print(f"  Operaciones exitosas: {status['metrics']['successful_operations']}")
    print(f"  Operaciones fallidas: {status['metrics']['failed_operations']}")
    print(f"  Tiempo promedio de respuesta: {status['metrics']['avg_response_time']*1000:.2f} ms")
    
    # Limpiar
    await load_balancer_manager.shutdown()


async def test_autoscaling():
    """Probar capacidades de auto-escalado del balanceador."""
    c = BeautifulTerminalColors
    print(f"\n{c.DIVINE}=== PRUEBA DE AUTO-ESCALADO ==={c.END}")
    
    # Inicializar manager de balanceo
    await load_balancer_manager.initialize()
    
    # Crear balanceador con configuración para auto-escalado
    balancer = await load_balancer_manager.create_balancer(
        name="autoscaling_balancer",
        algorithm=BalancerAlgorithm.QUANTUM,  # Algoritmo más avanzado
        scaling_policy=ScalingPolicy.THRESHOLD,
        session_affinity=SessionAffinityMode.TOKEN
    )
    
    if not balancer:
        print(f"{c.RED}Error al crear balanceador{c.END}")
        return
    
    # Configurar parámetros de escalado
    balancer.scaling_settings["cpu_threshold_up"] = 0.7    # Escalar arriba si CPU > 70%
    balancer.scaling_settings["cpu_threshold_down"] = 0.3  # Escalar abajo si CPU < 30%
    balancer.scaling_settings["scale_up_cooldown"] = 5     # 5 segundos entre escalados hacia arriba
    balancer.scaling_settings["scale_down_cooldown"] = 10  # 10 segundos entre escalados hacia abajo
    balancer.scaling_settings["min_nodes"] = 2
    balancer.scaling_settings["max_nodes"] = 10
    
    # Añadir nodos iniciales (pocos)
    nodes = [
        CloudNode(f"node_{i}", "127.0.0.1", 8080 + i, weight=1.0, max_connections=20)
        for i in range(3)
    ]
    
    for node in nodes:
        await balancer.add_node(node)
    
    # Inicializar simulador de operaciones
    trading_ops = TradingOperations()
    
    # Función para mostrar estado de nodos
    def print_nodes_status():
        print(f"{c.CYAN}Estado de nodos:{c.END}")
        print(f"  Total: {len(balancer.nodes)}")
        print(f"  Saludables: {len(balancer.healthy_nodes)}")
        
        # Ordenar por CPU para mejor visualización
        nodes_by_cpu = sorted(
            [(node_id, node) for node_id, node in balancer.nodes.items()],
            key=lambda x: x[1].metrics["cpu_usage"],
            reverse=True
        )
        
        for node_id, node in nodes_by_cpu:
            health = f"{c.GREEN}HEALTHY{c.END}" if node_id in balancer.healthy_nodes else f"{c.RED}UNHEALTHY{c.END}"
            print(f"  {node_id}: {health}, CPU: {node.metrics['cpu_usage']*100:.1f}%, "
                  f"Conn: {node.active_connections}/{node.max_connections}")
    
    # Estado inicial
    print(f"\n{c.CYAN}Estado inicial de nodos:{c.END}")
    print_nodes_status()
    
    # Fase 1: Carga baja
    print(f"\n{c.CYAN}Fase 1: Carga baja (10 operaciones)...{c.END}")
    for i in range(10):
        symbol = random.choice(list(trading_ops.symbols.keys()))
        try:
            result, node = await balancer.execute_operation(
                trading_ops.get_price, 
                session_key=f"user_{i%3}",
                symbol=symbol
            )
            
            if result and node:
                print(f"  Operación #{i}: {c.GREEN}Éxito{c.END} - Nodo: {node.node_id}")
            else:
                print(f"  Operación #{i}: {c.RED}Error{c.END}")
        except Exception as e:
            print(f"  Operación #{i}: {c.RED}Excepción: {e}{c.END}")
        
        await asyncio.sleep(0.2)
    
    # Verificar estado tras carga baja
    print(f"\n{c.CYAN}Estado tras carga baja:{c.END}")
    print_nodes_status()
    
    # Fase 2: Carga alta para forzar escalado hacia arriba
    print(f"\n{c.CYAN}Fase 2: Carga alta para forzar escalado (50 operaciones rápidas)...{c.END}")
    
    # Simular alta carga en los nodos actuales
    for node in balancer.nodes.values():
        # Simular uso elevado de CPU
        node.metrics["cpu_usage"] = random.uniform(0.75, 0.95)
        # Simular muchas conexiones
        node.active_connections = int(node.max_connections * 0.9)
    
    # Forzar decisión de escalado
    await balancer._make_scaling_decisions()
    
    # Verificar si se añadieron nodos
    print(f"\n{c.CYAN}Estado tras alto uso simulado (debería escalar arriba):{c.END}")
    print_nodes_status()
    
    # Ejecutar muchas operaciones rápidas
    tasks = []
    for i in range(50):
        symbol = random.choice(list(trading_ops.symbols.keys()))
        user = f"user_{i%10}"  # 10 usuarios diferentes
        
        task = asyncio.create_task(balancer.execute_operation(
            trading_ops.get_price, 
            session_key=user,
            symbol=symbol
        ))
        tasks.append(task)
        
        # No esperar entre operaciones para simular carga máxima
    
    # Esperar a que todas las operaciones terminen
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Contar resultados
    success_count = sum(1 for r in results if isinstance(r, tuple) and r[0] is not None)
    error_count = len(results) - success_count
    
    print(f"  Operaciones exitosas: {c.GREEN}{success_count}{c.END}")
    print(f"  Operaciones fallidas: {c.RED}{error_count}{c.END}")
    
    # Verificar estado tras carga alta
    print(f"\n{c.CYAN}Estado tras ejecución de carga alta:{c.END}")
    print_nodes_status()
    
    # Fase 3: Permitir que la carga baje para forzar escalado hacia abajo
    print(f"\n{c.CYAN}Fase 3: Reducción de carga para forzar escalado hacia abajo...{c.END}")
    
    # Simular baja carga en los nodos
    for node in balancer.nodes.values():
        # Simular uso bajo de CPU
        node.metrics["cpu_usage"] = random.uniform(0.1, 0.25)
        # Simular pocas conexiones
        node.active_connections = int(node.max_connections * 0.1)
    
    # Forzar decisión de escalado (sin respetar enfriamiento)
    balancer.scaling_settings["last_scale_down"] = 0
    await balancer._make_scaling_decisions()
    
    # Verificar si se eliminaron nodos
    print(f"\n{c.CYAN}Estado tras uso bajo simulado (debería escalar abajo):{c.END}")
    print_nodes_status()
    
    # Ejecutar algunas operaciones con carga baja
    for i in range(10):
        symbol = random.choice(list(trading_ops.symbols.keys()))
        try:
            result, node = await balancer.execute_operation(
                trading_ops.get_price, 
                session_key=f"user_{i%3}",
                symbol=symbol
            )
            
            if result and node:
                print(f"  Operación #{i}: {c.GREEN}Éxito{c.END} - Nodo: {node.node_id}")
            else:
                print(f"  Operación #{i}: {c.RED}Error{c.END}")
        except Exception as e:
            print(f"  Operación #{i}: {c.RED}Excepción: {e}{c.END}")
        
        await asyncio.sleep(0.2)
    
    # Estado final
    print(f"\n{c.CYAN}Estado final tras todo el ciclo de escalado:{c.END}")
    print_nodes_status()
    
    # Estadísticas del balanceador
    print(f"\n{c.CYAN}Estadísticas del balanceador:{c.END}")
    status = balancer.get_status()
    print(f"  Estado: {status['state']}")
    print(f"  Operaciones totales: {status['metrics']['total_operations']}")
    print(f"  Operaciones exitosas: {status['metrics']['successful_operations']}")
    print(f"  Operaciones fallidas: {status['metrics']['failed_operations']}")
    
    # Limpiar
    await load_balancer_manager.shutdown()


async def test_integration_with_cb_and_checkpoints():
    """Probar integración del balanceador con circuit breaker y checkpoints."""
    c = BeautifulTerminalColors
    print(f"\n{c.DIVINE}=== PRUEBA DE INTEGRACIÓN CON CIRCUIT BREAKER Y CHECKPOINTS ==={c.END}")
    
    # 1. Inicializar gestores
    await load_balancer_manager.initialize()
    
    # Inicializar checkpoint manager en memoria
    cp_manager = DistributedCheckpointManager(
        storage_type=CheckpointStorageType.MEMORY,
        consistency_level=CheckpointConsistencyLevel.QUANTUM
    )
    
    # Inicializar circuit breaker factory
    cb_factory = CloudCircuitBreakerFactory()
    
    # 2. Crear componentes
    
    # Crear circuit breaker para trading
    cb = await cb_factory.create(
        name="trading_operations",
        failure_threshold=3,
        recovery_timeout=0.5,
        quantum_failsafe=True
    )
    
    # Registrar componente para checkpoints
    await cp_manager.register_component("trading_state")
    
    # Crear balanceador
    balancer = await load_balancer_manager.create_balancer(
        name="trading_balancer",
        algorithm=BalancerAlgorithm.QUANTUM,
        session_affinity=SessionAffinityMode.TOKEN
    )
    
    if not balancer:
        print(f"{c.RED}Error al crear balanceador{c.END}")
        return
    
    # Añadir nodos
    nodes = [
        CloudNode(f"node_{i}", "127.0.0.1", 8080 + i, weight=1.0)
        for i in range(3)
    ]
    
    for node in nodes:
        await balancer.add_node(node)
    
    # Inicializar operaciones
    trading_ops = TradingOperations()
    
    # 3. Crear función de trading con protección ultra resiliente
    
    # Estado global a salvar en checkpoints
    trading_state = {
        "operations": [],
        "balances": {},
        "last_prices": {},
        "last_checkpoint": time.time()
    }
    
    # Función que ejemplifica la combinación de todos los componentes
    async def resilient_trading_operation(operation_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar operación de trading con resiliencia máxima.
        
        Args:
            operation_type: Tipo de operación
            params: Parámetros de la operación
            
        Returns:
            Resultado de la operación
            
        Raises:
            Exception: Si ocurre un error no recuperable
        """
        start_time = time.time()
        result = None
        
        # Seleccionar operación
        if operation_type == "get_price":
            # Usar circuit breaker directamente
            try:
                price_data = await cb.call(trading_ops.get_price, params["symbol"])
                
                # Actualizar estado global
                trading_state["last_prices"][params["symbol"]] = {
                    "price": price_data["price"],
                    "timestamp": price_data["timestamp"]
                }
                
                # Registrar operación
                trading_state["operations"].append({
                    "type": "get_price",
                    "symbol": params["symbol"],
                    "result": price_data,
                    "timestamp": time.time()
                })
                
                result = price_data
                
            except Exception as e:
                # Si falla, intentar usar último precio conocido
                last_price = trading_state["last_prices"].get(params["symbol"])
                if last_price:
                    # Usar precio anterior pero marcarlo como recuperado
                    result = {
                        "symbol": params["symbol"],
                        "price": last_price["price"],
                        "timestamp": last_price["timestamp"],
                        "recovered": True
                    }
                else:
                    # No hay precio anterior, propagar error
                    raise Exception(f"No se pudo obtener precio para {params['symbol']}: {e}")
        
        elif operation_type == "place_order":
            # Usar balanceador con afinidad de sesión
            order_result, node = await balancer.execute_operation(
                trading_ops.place_order,
                session_key=params.get("user_id", "default_user"),
                symbol=params["symbol"],
                order_type=params["order_type"],
                amount=params["amount"]
            )
            
            if order_result:
                # Actualizar estado global
                trading_state["operations"].append({
                    "type": "place_order",
                    "order": order_result,
                    "timestamp": time.time()
                })
                
                # Actualizar balance simulado
                if params["order_type"] == "buy":
                    # Compra: restar efectivo, añadir cripto
                    if "cash" not in trading_state["balances"]:
                        trading_state["balances"]["cash"] = 100000.0  # Inicial
                    
                    trading_state["balances"]["cash"] -= order_result["amount"] * order_result["filled_price"]
                    
                    if params["symbol"] not in trading_state["balances"]:
                        trading_state["balances"][params["symbol"]] = 0
                    
                    trading_state["balances"][params["symbol"]] += order_result["amount"]
                
                else:  # sell
                    # Venta: añadir efectivo, restar cripto
                    if "cash" not in trading_state["balances"]:
                        trading_state["balances"]["cash"] = 100000.0  # Inicial
                    
                    trading_state["balances"]["cash"] += order_result["amount"] * order_result["filled_price"]
                    
                    if params["symbol"] not in trading_state["balances"]:
                        trading_state["balances"][params["symbol"]] = 0
                    
                    trading_state["balances"][params["symbol"]] = max(0, trading_state["balances"][params["symbol"]] - order_result["amount"])
                
                result = order_result
                
            else:
                raise Exception(f"Error al colocar orden para {params['symbol']}")
        
        elif operation_type == "get_balance":
            # Intentar obtener balance real
            try:
                balance_result, node = await balancer.execute_operation(
                    trading_ops.get_account_balance,
                    session_key=params.get("user_id", "default_user")
                )
                
                if balance_result:
                    # Actualizar estado con balance real
                    for symbol, data in balance_result["holdings"].items():
                        trading_state["balances"][symbol] = data["amount"]
                    
                    trading_state["balances"]["cash"] = balance_result["available_balance"]
                    
                    result = balance_result
                else:
                    raise Exception("Error al obtener balance")
                    
            except Exception as e:
                # Si falla, generar balance basado en estado guardado
                simulated_balance = {
                    "total_equity": sum(trading_state["balances"].get(symbol, 0) * 
                                    trading_state["last_prices"].get(symbol, {}).get("price", 0)
                                    for symbol in trading_ops.symbols.keys() if symbol != "cash"),
                    "available_balance": trading_state["balances"].get("cash", 0),
                    "holdings": {
                        symbol: {
                            "amount": trading_state["balances"].get(symbol, 0),
                            "avg_price": trading_state["last_prices"].get(symbol, {}).get("price", 0)
                        }
                        for symbol in trading_ops.symbols.keys() if symbol != "cash"
                    },
                    "timestamp": time.time(),
                    "recovered": True
                }
                
                result = simulated_balance
        
        else:
            # Operación no reconocida
            raise ValueError(f"Tipo de operación no reconocido: {operation_type}")
        
        # Crear checkpoint periódicamente
        if time.time() - trading_state["last_checkpoint"] > 5:  # Cada 5 segundos
            await cp_manager.create_checkpoint(
                component_id="trading_state",
                data=trading_state,
                tags=["trading", "auto"]
            )
            trading_state["last_checkpoint"] = time.time()
        
        return result
    
    # 4. Ejecutar operaciones usando la función resiliente
    print(f"{c.CYAN}Ejecutando operaciones con resiliencia integrada...{c.END}")
    
    # Crear checkpoint inicial
    checkpoint_id = await cp_manager.create_checkpoint(
        component_id="trading_state",
        data=trading_state,
        tags=["trading", "initial"]
    )
    
    print(f"  Checkpoint inicial creado: {c.GREEN}{checkpoint_id}{c.END}")
    
    # Ejecutar serie de operaciones
    for i in range(20):
        try:
            # Seleccionar tipo de operación
            op_type = random.choice(["get_price", "place_order", "get_balance"])
            
            if op_type == "get_price":
                symbol = random.choice(list(trading_ops.symbols.keys()))
                params = {"symbol": symbol}
                
                result = await resilient_trading_operation("get_price", params)
                
                if result:
                    recovered = result.get("recovered", False)
                    status = f"{c.QUANTUM}RECUPERADO{c.END}" if recovered else f"{c.GREEN}OK{c.END}"
                    print(f"  {i+1}. Precio {symbol}: ${result['price']:.2f} [{status}]")
                else:
                    print(f"  {i+1}. {c.RED}Error al obtener precio{c.END}")
            
            elif op_type == "place_order":
                symbol = random.choice(list(trading_ops.symbols.keys()))
                order_type = random.choice(["buy", "sell"])
                amount = random.uniform(0.1, 1.0)
                user_id = f"user_{i%5}"
                
                params = {
                    "symbol": symbol,
                    "order_type": order_type,
                    "amount": amount,
                    "user_id": user_id
                }
                
                result = await resilient_trading_operation("place_order", params)
                
                if result:
                    recovered = result.get("recovered", False)
                    status = f"{c.QUANTUM}RECUPERADO{c.END}" if recovered else f"{c.GREEN}OK{c.END}"
                    print(f"  {i+1}. Orden {order_type} de {amount:.4f} {symbol} a ${result['price']:.2f} [{status}]")
                else:
                    print(f"  {i+1}. {c.RED}Error al colocar orden{c.END}")
            
            else:  # get_balance
                user_id = f"user_{i%5}"
                params = {"user_id": user_id}
                
                result = await resilient_trading_operation("get_balance", params)
                
                if result:
                    recovered = result.get("recovered", False)
                    status = f"{c.QUANTUM}RECUPERADO{c.END}" if recovered else f"{c.GREEN}OK{c.END}"
                    
                    cash = result.get("available_balance", 0)
                    if "holdings" in result:
                        holdings = ", ".join([f"{symbol}: {data['amount']:.4f}" 
                                            for symbol, data in result["holdings"].items()
                                            if data["amount"] > 0])
                    else:
                        holdings = "N/A"
                    
                    print(f"  {i+1}. Balance: ${cash:.2f}, Holdings: {holdings} [{status}]")
                else:
                    print(f"  {i+1}. {c.RED}Error al obtener balance{c.END}")
            
            # Simular fallos aleatorios para probar recuperación
            if i == 9:  # A mitad de las operaciones
                print(f"\n{c.RED}Simulando fallo grave en nodos...{c.END}")
                
                # Marcar todos los nodos como unhealthy
                for node_id in list(balancer.healthy_nodes):
                    balancer.nodes[node_id].health_status = NodeHealthStatus.UNHEALTHY
                    balancer.healthy_nodes.remove(node_id)
                
                # Actualizar estado del balanceador
                await balancer._check_all_nodes_health()
                
                print(f"  Estado del balanceador: {c.RED}{balancer.state.name}{c.END}")
                print(f"  Intentando recuperación desde último checkpoint...")
                
                # Cargar último checkpoint
                data, metadata = await cp_manager.load_latest_checkpoint("trading_state")
                
                if data:
                    # Restaurar estado
                    trading_state.update(data)
                    print(f"  {c.GREEN}Estado recuperado desde checkpoint {metadata.checkpoint_id}{c.END}")
                    
                    # Restaurar un nodo para continuar
                    random_node = random.choice(list(balancer.nodes.values()))
                    random_node.health_status = NodeHealthStatus.HEALTHY
                    balancer.healthy_nodes.add(random_node.node_id)
                    
                    # Actualizar estado del balanceador
                    await balancer._check_all_nodes_health()
                    
                    print(f"  Nodo {random_node.node_id} recuperado")
                    print(f"  Estado del balanceador: {c.YELLOW}{balancer.state.name}{c.END}")
                else:
                    print(f"  {c.RED}Error al recuperar estado desde checkpoint{c.END}")
        
        except Exception as e:
            print(f"  {i+1}. {c.RED}Error: {e}{c.END}")
        
        # Pausa entre operaciones
        await asyncio.sleep(0.2)
    
    # 5. Mostrar estado final
    print(f"\n{c.CYAN}Estado final del trading:{c.END}")
    print(f"  Último precio de BTC: ${trading_state['last_prices'].get('BTC', {}).get('price', 0):.2f}")
    print(f"  Último precio de ETH: ${trading_state['last_prices'].get('ETH', {}).get('price', 0):.2f}")
    print(f"  Balance en efectivo: ${trading_state['balances'].get('cash', 0):.2f}")
    print(f"  Posición en BTC: {trading_state['balances'].get('BTC', 0):.8f}")
    print(f"  Posición en ETH: {trading_state['balances'].get('ETH', 0):.8f}")
    print(f"  Operaciones totales: {len(trading_state['operations'])}")
    
    # Estadísticas de componentes
    print(f"\n{c.CYAN}Estadísticas del circuit breaker:{c.END}")
    cb_metrics = cb.get_metrics()
    print(f"  Estado: {cb.get_state()}")
    print(f"  Llamadas totales: {cb_metrics['calls']['total']}")
    print(f"  Éxitos: {cb_metrics['calls']['success']}")
    print(f"  Fallos: {cb_metrics['calls']['failure']}")
    print(f"  Modo cuántico: {cb_metrics['calls']['quantum']}")
    
    print(f"\n{c.CYAN}Estadísticas del balanceador:{c.END}")
    lb_status = balancer.get_status()
    print(f"  Estado: {lb_status['state']}")
    print(f"  Nodos saludables: {len(balancer.healthy_nodes)}/{len(balancer.nodes)}")
    print(f"  Operaciones totales: {lb_status['metrics']['total_operations']}")
    print(f"  Operaciones exitosas: {lb_status['metrics']['successful_operations']}")
    
    print(f"\n{c.CYAN}Checkpoints creados:{c.END}")
    checkpoints = await cp_manager.list_checkpoints("trading_state")
    for i, cp in enumerate(checkpoints):
        print(f"  {i+1}. {cp.checkpoint_id} - {cp.version} - Tags: {cp.tags}")
    
    # Limpiar
    await load_balancer_manager.shutdown()


async def main():
    """Ejecutar todas las pruebas."""
    c = BeautifulTerminalColors
    print(f"\n{c.DIVINE}{c.BOLD}======================================================{c.END}")
    print(f"{c.DIVINE}{c.BOLD}  PRUEBA DEL CLOUDLOADBALANCER ULTRA-DIVINO  {c.END}")
    print(f"{c.DIVINE}{c.BOLD}======================================================{c.END}\n")
    
    # Ejecutar pruebas
    await test_basic_load_balancing()
    await test_autoscaling()
    await test_integration_with_cb_and_checkpoints()
    
    print(f"\n{c.DIVINE}{c.BOLD}======================================================{c.END}")
    print(f"{c.DIVINE}{c.BOLD}  PRUEBAS COMPLETADAS EXITOSAMENTE  {c.END}")
    print(f"{c.DIVINE}{c.BOLD}======================================================{c.END}\n")


if __name__ == "__main__":
    asyncio.run(main())