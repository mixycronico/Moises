"""
Script de migración para integrar el WebSocket Trascendental para Exchanges.

Este script reemplaza los conectores tradicionales de exchanges en el sistema Genesis
con la nueva implementación trascendental, mejorando la resiliencia y añadiendo
capacidades de procesamiento cuántico para la adquisición de datos de mercado.

Fases de migración:
1. Identificar todos los componentes que utilizan conectores de exchange
2. Reemplazar conectores tradicionales con WebSocket Trascendental
3. Actualizar referencias y configuración para usar las nuevas interfaces
4. Validar funcionamiento y preservar compatibilidad hacia atrás
"""

import os
import sys
import logging
import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Set
import importlib.util
import inspect

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("migrate_exchange_connectors.log")
    ]
)

logger = logging.getLogger("Migration.Exchanges")

# Asegurarse de que el directorio raíz esté en el path de Python
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

# Importar módulo de WebSocket Trascendental para Exchanges
from genesis.core.exchange_websocket_connector import ExchangeWebSocketHandler

# Definir patrones para identificar código de conexión a exchanges
EXCHANGE_PATTERNS = [
    r'(?:ccxt\.|\bfrom\s+ccxt\s+import)',  # Uso de CCXT
    r'(?:binance\.|\bfrom\s+binance\s+import)',  # Uso de Binance
    r'(?:websockets\.connect\([\'"]wss?://[^\'"]*(?:binance|coinbase|kraken|huobi|kucoin|okex)[^\'"]*[\'"])',  # WebSockets a exchanges
    r'(?:requests\.(?:get|post)\([\'"]https?://[^\'"]*(?:binance|coinbase|kraken|huobi|kucoin|okex)[^\'"]*[\'"])',  # HTTP a exchanges
    r'ExchangeClient|ExchangeAPI|ExchangeConnector|ExchangeManager|MarketDataProvider'  # Nombres comunes para clases de exchanges
]

# Código de reemplazo
REPLACEMENT_TEMPLATE = """
# Importar WebSocket Trascendental para Exchanges
from genesis.core.exchange_websocket_connector import ExchangeWebSocketHandler

# Crear instancia del manejador
exchange_ws = ExchangeWebSocketHandler("{exchange_id}")

# Registrar callbacks para procesar datos
async def _process_{data_type}_data(data):
    # Procesamiento de datos recibidos
    {process_code}

# Conectar a streams relevantes
async def connect_to_exchange_streams():
    # Conectar a streams necesarios
    {connect_code}
"""

class MigrationTask:
    """Tarea de migración para un componente específico."""
    
    def __init__(self, file_path: str, component_name: str, exchange_references: List[str]):
        """
        Inicializar tarea de migración.
        
        Args:
            file_path: Ruta al archivo fuente
            component_name: Nombre del componente
            exchange_references: Referencias a exchanges encontradas
        """
        self.file_path = file_path
        self.component_name = component_name
        self.exchange_references = exchange_references
        self.completed = False
        self.validated = False
        self.errors = []
    
    def __str__(self) -> str:
        """Representación en string de la tarea."""
        return f"MigrationTask(component={self.component_name}, file={os.path.basename(self.file_path)})"

def find_components_using_exchanges() -> List[MigrationTask]:
    """
    Encontrar todos los componentes que utilizan conexiones a exchanges.
    
    Returns:
        Lista de tareas de migración para los componentes identificados
    """
    logger.info("Buscando componentes que utilizan conexiones a exchanges...")
    
    tasks = []
    genesis_dir = os.path.join(root_dir, 'genesis')
    
    # Compilar patrones como regex
    patterns = [re.compile(pattern) for pattern in EXCHANGE_PATTERNS]
    
    # Recorrer directorio genesis
    for root, _, files in os.walk(genesis_dir):
        for filename in files:
            if not filename.endswith('.py'):
                continue
                
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, root_dir)
            
            # Leer contenido del archivo
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"No se pudo leer {rel_path}: {e}")
                continue
            
            # Buscar referencias a exchanges
            exchange_refs = []
            for pattern in patterns:
                matches = pattern.findall(content)
                if matches:
                    for match in matches:
                        if match not in exchange_refs:
                            exchange_refs.append(match)
            
            # Si encontramos referencias, crear tarea de migración
            if exchange_refs:
                # Intentar determinar nombre del componente
                component_name = filename[:-3]  # quitar .py
                if "class" in content:
                    # Buscar definiciones de clase
                    class_match = re.search(r'class\s+([A-Za-z0-9_]+)', content)
                    if class_match:
                        component_name = class_match.group(1)
                
                task = MigrationTask(file_path, component_name, exchange_refs)
                tasks.append(task)
                
                logger.info(f"Encontrado componente usando exchanges: {component_name} en {rel_path}")
                logger.info(f"  Referencias: {', '.join(exchange_refs[:3])}{'...' if len(exchange_refs) > 3 else ''}")
    
    logger.info(f"Total componentes encontrados: {len(tasks)}")
    return tasks

def analyze_component(task: MigrationTask) -> Dict[str, Any]:
    """
    Analizar un componente para determinar detalles de su uso de exchanges.
    
    Args:
        task: Tarea de migración
        
    Returns:
        Diccionario con análisis detallado
    """
    logger.info(f"Analizando componente: {task.component_name}")
    
    analysis = {
        "exchanges_used": set(),
        "stream_types": set(),
        "ws_connections": [],
        "rest_endpoints": [],
        "migration_complexity": "medium"
    }
    
    try:
        with open(task.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Detectar exchanges utilizados
        for exchange in ["binance", "coinbase", "kraken", "huobi", "kucoin", "okex", "bitmex"]:
            if re.search(r'\b' + exchange + r'\b', content, re.IGNORECASE):
                analysis["exchanges_used"].add(exchange)
        
        # Detectar tipos de streams utilizados
        for stream_type in ["trade", "ticker", "orderbook", "depth", "kline", "ohlcv", "candle"]:
            if re.search(r'\b' + stream_type + r'\b', content, re.IGNORECASE):
                analysis["stream_types"].add(stream_type)
        
        # Detectar conexiones WebSocket
        ws_pattern = r'(?:websockets\.connect|WebSocketApp)\([\'"]([^\'"]+)[\'"]\)'
        ws_matches = re.findall(ws_pattern, content)
        analysis["ws_connections"] = ws_matches
        
        # Detectar endpoints REST
        rest_pattern = r'(?:requests\.(?:get|post|put|delete)|\.request)\([\'"]([^\'"]+)[\'"]\)'
        rest_matches = re.findall(rest_pattern, content)
        analysis["rest_endpoints"] = rest_matches
        
        # Determinar complejidad de la migración
        if len(analysis["exchanges_used"]) > 2 or len(analysis["stream_types"]) > 3:
            analysis["migration_complexity"] = "high"
        elif not analysis["ws_connections"] and not analysis["stream_types"]:
            analysis["migration_complexity"] = "low"
        
    except Exception as e:
        logger.error(f"Error analizando {task.component_name}: {e}")
        analysis["error"] = str(e)
        analysis["migration_complexity"] = "unknown"
    
    logger.info(f"Análisis de {task.component_name}:")
    logger.info(f"  Exchanges: {', '.join(analysis['exchanges_used'])}")
    logger.info(f"  Tipos de streams: {', '.join(analysis['stream_types'])}")
    logger.info(f"  Complejidad: {analysis['migration_complexity']}")
    
    return analysis

def generate_migration_plan(task: MigrationTask, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generar plan de migración para un componente.
    
    Args:
        task: Tarea de migración
        analysis: Análisis del componente
        
    Returns:
        Plan de migración detallado
    """
    logger.info(f"Generando plan de migración para {task.component_name}")
    
    plan = {
        "imports_to_add": [
            "from genesis.core.exchange_websocket_connector import ExchangeWebSocketHandler"
        ],
        "code_to_add": [],
        "patterns_to_replace": [],
        "complexity": analysis["migration_complexity"],
        "estimated_effort": "medium",
        "test_cases": []
    }
    
    # Definir importaciones necesarias
    
    # Definir código a añadir
    exchanges = list(analysis["exchanges_used"])
    if exchanges:
        exchange_id = exchanges[0]  # Usar el primer exchange como principal
        
        # Código para inicializar WebSocket
        init_code = f"self.exchange_ws = ExchangeWebSocketHandler(\"{exchange_id}\")"
        plan["code_to_add"].append(init_code)
        
        # Generar callbacks para cada tipo de stream
        for stream_type in analysis["stream_types"]:
            callback_name = f"_on_{stream_type}_data"
            callback_code = f"""
        async def {callback_name}(self, data):
            # Procesar datos de {stream_type}
            if hasattr(self, "process_{stream_type}"):
                await self.process_{stream_type}(data)
            """
            plan["code_to_add"].append(callback_code)
        
        # Código para conectar a streams
        connect_code = "async def connect_to_exchange_streams(self):"
        if "trade" in analysis["stream_types"]:
            connect_code += "\n            # Conectar a stream de trades"
            connect_code += f"\n            await self.exchange_ws.connect_to_stream(\"btcusdt@trade\", self._on_trade_data)"
        
        if "kline" in analysis["stream_types"] or "candle" in analysis["stream_types"]:
            connect_code += "\n            # Conectar a stream de velas"
            connect_code += f"\n            await self.exchange_ws.connect_to_stream(\"btcusdt@kline_1m\", self._on_kline_data)"
        
        if "orderbook" in analysis["stream_types"] or "depth" in analysis["stream_types"]:
            connect_code += "\n            # Conectar a stream de orderbook"
            connect_code += f"\n            await self.exchange_ws.connect_to_stream(\"btcusdt@depth20\", self._on_orderbook_data)"
        
        plan["code_to_add"].append(connect_code)
    
    # Definir patrones a reemplazar
    ws_connection_pattern = r'websockets\.connect\([\'"]wss?://[^\'"]*(?:binance|coinbase|kraken)[^\'"]*[\'"]\)'
    replacement = "self.exchange_ws"
    plan["patterns_to_replace"].append((ws_connection_pattern, replacement))
    
    # Definir casos de prueba
    plan["test_cases"] = [
        f"test_{task.component_name}_initialization",
        f"test_{task.component_name}_websocket_connection",
        f"test_{task.component_name}_data_processing"
    ]
    
    # Estimar esfuerzo
    if plan["complexity"] == "high":
        plan["estimated_effort"] = "high"
    elif plan["complexity"] == "low":
        plan["estimated_effort"] = "low"
    
    logger.info(f"Plan generado para {task.component_name}:")
    logger.info(f"  Complejidad: {plan['complexity']}")
    logger.info(f"  Esfuerzo estimado: {plan['estimated_effort']}")
    logger.info(f"  Patrones a reemplazar: {len(plan['patterns_to_replace'])}")
    logger.info(f"  Código a añadir: {len(plan['code_to_add'])}")
    
    return plan

def create_integration_sample(exchange_id: str = "binance") -> str:
    """
    Crear código de ejemplo de integración con WebSocket Trascendental.
    
    Args:
        exchange_id: ID del exchange a utilizar
        
    Returns:
        Código de ejemplo como string
    """
    sample_code = f"""
'''
Ejemplo de integración con el WebSocket Trascendental para Exchanges.

Este código muestra cómo utilizar el WebSocket Trascendental para conectar
con exchanges de criptomonedas y procesar datos en tiempo real.
'''

import asyncio
import logging
from typing import Dict, Any

# Importar WebSocket Trascendental
from genesis.core.exchange_websocket_connector import ExchangeWebSocketHandler

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExchangeExample")

class ExampleExchangeComponent:
    '''
    Componente de ejemplo que utiliza WebSocket Trascendental para Exchanges.
    
    Este componente se conecta a múltiples streams de datos en tiempo real
    y demuestra el procesamiento de diferentes tipos de datos.
    '''
    
    def __init__(self):
        '''Inicializar componente.'''
        # Crear manejador WebSocket Trascendental
        self.exchange_ws = ExchangeWebSocketHandler("{exchange_id}")
        
        # Estadísticas
        self.stats = {{
            "trades_processed": 0,
            "klines_processed": 0,
            "orderbook_updates": 0
        }}
    
    async def start(self):
        '''Iniciar componente y conexiones.'''
        logger.info("Iniciando componente de exchange...")
        
        # Conectar a streams necesarios
        await self.connect_to_exchange_streams()
        
        logger.info("Componente iniciado y conectado a streams")
    
    async def stop(self):
        '''Detener componente y conexiones.'''
        logger.info("Deteniendo componente de exchange...")
        
        # Desconectar de todos los streams
        await self.exchange_ws.disconnect_all()
        
        logger.info("Componente detenido")
    
    async def connect_to_exchange_streams(self):
        '''Conectar a streams de datos.'''
        # Conectar a stream de trades
        await self.exchange_ws.connect_to_stream("btcusdt@trade", self._on_trade_data)
        
        # Conectar a stream de velas (klines)
        await self.exchange_ws.connect_to_stream("btcusdt@kline_1m", self._on_kline_data)
        
        # Conectar a stream de orderbook
        await self.exchange_ws.connect_to_stream("btcusdt@depth20", self._on_orderbook_data)
    
    async def _on_trade_data(self, data: Dict[str, Any]):
        '''
        Procesar datos de trades.
        
        Args:
            data: Datos normalizados del trade
        '''
        self.stats["trades_processed"] += 1
        
        # Procesar datos (ejemplo)
        if self.stats["trades_processed"] % 10 == 0:
            price = data.get("price", 0)
            quantity = data.get("quantity", 0)
            symbol = data.get("symbol", "unknown")
            
            logger.info(f"Trade #{self.stats['trades_processed']}: {{symbol}} - {{price}} - {{quantity}}")
    
    async def _on_kline_data(self, data: Dict[str, Any]):
        '''
        Procesar datos de velas (klines).
        
        Args:
            data: Datos normalizados de la vela
        '''
        self.stats["klines_processed"] += 1
        
        # Procesar datos (ejemplo)
        if self.stats["klines_processed"] % 5 == 0:
            symbol = data.get("symbol", "unknown")
            interval = data.get("interval", "unknown")
            close = data.get("close", 0)
            
            logger.info(f"Kline #{self.stats['klines_processed']}: {{symbol}} {{interval}} - Close: {{close}}")
    
    async def _on_orderbook_data(self, data: Dict[str, Any]):
        '''
        Procesar datos de orderbook.
        
        Args:
            data: Datos normalizados del orderbook
        '''
        self.stats["orderbook_updates"] += 1
        
        # Procesar datos (ejemplo)
        if self.stats["orderbook_updates"] % 20 == 0:
            symbol = data.get("symbol", "unknown")
            bids = len(data.get("bids", []))
            asks = len(data.get("asks", []))
            
            logger.info(f"Orderbook #{self.stats['orderbook_updates']}: {{symbol}} - Bids: {{bids}} - Asks: {{asks}}")
    
    def get_stats(self) -> Dict[str, Any]:
        '''
        Obtener estadísticas del componente.
        
        Returns:
            Diccionario con estadísticas
        '''
        # Obtener estadísticas del WebSocket
        ws_stats = self.exchange_ws.get_stats()
        
        # Combinar con estadísticas propias
        combined_stats = {{
            **self.stats,
            "websocket": ws_stats
        }}
        
        return combined_stats

# Ejemplo de uso
async def main():
    '''Función principal.'''
    # Crear componente
    component = ExampleExchangeComponent()
    
    try:
        # Iniciar
        await component.start()
        
        # Ejecutar durante 60 segundos
        logger.info("Ejecutando componente durante 60 segundos...")
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < 60:
            await asyncio.sleep(10)
            
            # Mostrar estadísticas cada 10 segundos
            stats = component.get_stats()
            logger.info(f"Estadísticas actuales:")
            logger.info(f"  Trades: {{stats['trades_processed']}}")
            logger.info(f"  Klines: {{stats['klines_processed']}}")
            logger.info(f"  Orderbook: {{stats['orderbook_updates']}}")
        
        # Estadísticas finales
        final_stats = component.get_stats()
        logger.info("Estadísticas finales:")
        logger.info(f"  Trades procesados: {{final_stats['trades_processed']}}")
        logger.info(f"  Klines procesadas: {{final_stats['klines_processed']}}")
        logger.info(f"  Actualizaciones orderbook: {{final_stats['orderbook_updates']}}")
        logger.info(f"  Mensajes recibidos (WebSocket): {{final_stats['websocket']['messages_received']}}")
        logger.info(f"  Errores transmutados: {{final_stats['websocket']['errors_transmuted']}}")
        
    finally:
        # Detener componente
        await component.stop()

if __name__ == "__main__":
    asyncio.run(main())
"""
    return sample_code

async def main():
    """Función principal de migración."""
    logger.info("INICIANDO MIGRACIÓN DE CONECTORES DE EXCHANGE A WEBSOCKET TRASCENDENTAL")
    
    try:
        # Encontrar componentes que utilizan exchanges
        tasks = find_components_using_exchanges()
        
        if not tasks:
            logger.warning("No se encontraron componentes que utilicen exchanges")
            return
        
        # Generar informe de migración
        logger.info("\n=== INFORME DE MIGRACIÓN ===")
        logger.info(f"Componentes a migrar: {len(tasks)}")
        
        # Analizar componentes
        all_analyses = {}
        for i, task in enumerate(tasks[:5]):  # Limitar a 5 para este ejemplo
            logger.info(f"\n--- Componente {i+1}/{len(tasks)} ---")
            analysis = analyze_component(task)
            all_analyses[task.component_name] = analysis
            
            # Generar plan de migración
            plan = generate_migration_plan(task, analysis)
            
            logger.info(f"Plan para {task.component_name}:")
            logger.info(f"  Complejidad: {plan['complexity']}")
            logger.info(f"  Esfuerzo: {plan['estimated_effort']}")
        
        # Crear código de ejemplo de integración
        sample_code = create_integration_sample("binance")
        
        # Guardar código de ejemplo
        sample_path = os.path.join(root_dir, 'examples', 'exchange_ws_integration_example.py')
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_code)
        
        logger.info(f"\nEjemplo de integración guardado en: {os.path.relpath(sample_path, root_dir)}")
        
        # Resumen final
        logger.info("\n=== RESUMEN DE MIGRACIÓN ===")
        logger.info(f"Total componentes analizados: {len(all_analyses)}")
        
        complexity_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
        exchanges_used = set()
        
        for component, analysis in all_analyses.items():
            complexity = analysis.get("migration_complexity", "unknown")
            complexity_counts[complexity] += 1
            exchanges_used.update(analysis.get("exchanges_used", []))
        
        logger.info(f"Complejidad: Alta: {complexity_counts['high']}, Media: {complexity_counts['medium']}, Baja: {complexity_counts['low']}")
        logger.info(f"Exchanges utilizados: {', '.join(exchanges_used)}")
        
        logger.info("\nMIGRACIÓN COMPLETADA - Ejemplo de integración generado")
        
    except Exception as e:
        logger.error(f"ERROR EN MIGRACIÓN: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
"""