#!/usr/bin/env python3
"""
Ejemplo de integración completa entre Seraphim Trading y Simulador de Exchange.

Este script demuestra un flujo completo de trading usando el Sistema Seraphim
con un exchange simulado, mostrando cómo se comunicarían las distintas partes
del sistema en un caso de uso real.

El flujo incluye:
1. Obtener lista de símbolos disponibles
2. Obtener datos de mercado para un símbolo
3. Colocar una orden de compra
4. Verificar la orden
5. Cancelar la orden

Este script puede ejecutarse directamente o usarse como guía para
integrar el simulador en una aplicación real.
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, Any, List, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("seraphim_simulator_integration")

class SeraphimSimulatorClient:
    """Cliente para interactuar con la API Seraphim-Simulador."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Inicializar cliente Seraphim.
        
        Args:
            base_url: URL base del servidor
        """
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        """Crear sesión HTTP al entrar en contexto."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cerrar sesión HTTP al salir del contexto."""
        if self.session:
            await self.session.close()
    
    async def get_available_symbols(self) -> List[str]:
        """
        Obtener lista de símbolos disponibles.
        
        Returns:
            Lista de símbolos
        """
        async with self.session.get(f"{self.base_url}/api/seraphim/market/symbols") as response:
            data = await response.json()
            
            if not data.get("success"):
                logger.error(f"Error al obtener símbolos: {data.get('error')}")
                return []
            
            return data.get("symbols", [])
    
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Obtener datos de mercado para un símbolo.
        
        Args:
            symbol: Símbolo del mercado (ej: BTC-USDT)
            
        Returns:
            Datos de mercado o None si hay error
        """
        normalized_symbol = symbol.replace("/", "-")
        async with self.session.get(f"{self.base_url}/api/seraphim/market/data/{normalized_symbol}") as response:
            data = await response.json()
            
            if not data.get("success"):
                logger.error(f"Error al obtener datos de mercado: {data.get('error')}")
                return None
            
            return data
    
    async def place_order(self, symbol: str, side: str, order_type: str, 
                        amount: float, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Colocar una orden de trading.
        
        Args:
            symbol: Símbolo del mercado (ej: BTC/USDT)
            side: Lado de la orden (buy/sell)
            order_type: Tipo de orden (market/limit)
            amount: Cantidad
            price: Precio (solo para órdenes limit)
            
        Returns:
            Resultado de la orden o None si hay error
        """
        payload = {
            "symbol": symbol.replace("/", "-"),
            "side": side,
            "type": order_type,
            "amount": amount
        }
        
        if price and order_type.lower() == "limit":
            payload["price"] = price
        
        async with self.session.post(
            f"{self.base_url}/api/seraphim/trade/place", 
            json=payload
        ) as response:
            data = await response.json()
            
            if not data.get("success"):
                logger.error(f"Error al colocar orden: {data.get('error')}")
                return None
            
            return data
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancelar una orden existente.
        
        Args:
            order_id: ID de la orden
            
        Returns:
            True si se canceló correctamente
        """
        async with self.session.post(
            f"{self.base_url}/api/seraphim/trade/cancel/{order_id}",
            json={}  # Enviar cuerpo vacío para que Flask lo procese correctamente
        ) as response:
            data = await response.json()
            
            if not data.get("success"):
                logger.error(f"Error al cancelar orden: {data.get('error')}")
                return False
            
            return True
    
    async def get_orders(self, symbol: Optional[str] = None, status: str = "all") -> List[Dict[str, Any]]:
        """
        Obtener órdenes existentes.
        
        Args:
            symbol: Símbolo opcional para filtrar
            status: Estado de las órdenes (open/closed/all)
            
        Returns:
            Lista de órdenes
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.replace("/", "-")
        if status:
            params["status"] = status
            
        async with self.session.get(
            f"{self.base_url}/api/seraphim/trades",
            params=params
        ) as response:
            data = await response.json()
            
            if not data.get("success"):
                logger.error(f"Error al obtener órdenes: {data.get('error')}")
                return []
            
            return data.get("orders", [])

async def run_integration_demo():
    """Ejecutar demostración de integración completa."""
    print("\n🌟 SISTEMA SERAPHIM: DEMOSTRACIÓN DE INTEGRACIÓN CON SIMULADOR 🌟\n")
    
    try:
        async with SeraphimSimulatorClient() as client:
            # 1. Obtener símbolos disponibles
            print("🔍 Obteniendo símbolos disponibles...")
            symbols = await client.get_available_symbols()
            if not symbols:
                print("❌ No se pudieron obtener símbolos. Terminando demostración.")
                return
                
            print(f"✅ Símbolos disponibles: {', '.join(symbols)}")
            
            # Seleccionar primer símbolo para el ejemplo
            symbol = symbols[0]
            print(f"\n🪙 Usando símbolo {symbol} para la demostración\n")
            
            # 2. Obtener datos de mercado
            print(f"📊 Obteniendo datos de mercado para {symbol}...")
            market_data = await client.get_market_data(symbol)
            if not market_data:
                print("❌ No se pudieron obtener datos de mercado. Terminando demostración.")
                return
                
            price = market_data.get("last_price", 0)
            bid = market_data.get("bid", 0)
            ask = market_data.get("ask", 0)
            
            print(f"✅ Datos de mercado para {symbol}:")
            print(f"   Último precio: {price}")
            print(f"   Mejor oferta de compra (bid): {bid}")
            print(f"   Mejor oferta de venta (ask): {ask}")
            
            # 3. Colocar orden de compra
            print(f"\n💰 Colocando orden de compra para {symbol}...")
            order_amount = 0.01  # Cantidad pequeña para ejemplo
            order_price = ask * 0.99  # Ligeramente por debajo del ask
            
            order_result = await client.place_order(
                symbol=symbol,
                side="buy",
                order_type="limit",
                amount=order_amount,
                price=order_price
            )
            
            if not order_result:
                print("❌ No se pudo colocar la orden. Terminando demostración.")
                return
                
            order_id = order_result.get("order_id")
            print(f"✅ Orden colocada correctamente con ID: {order_id}")
            
            # 4. Verificar órdenes
            print(f"\n📋 Verificando órdenes abiertas...")
            # Esperar un momento para que la orden se registre
            await asyncio.sleep(1)
            
            orders = await client.get_orders(status="open")
            print(f"✅ Órdenes abiertas: {json.dumps(orders, indent=2)}")
            
            # 5. Cancelar orden
            print(f"\n🚫 Cancelando orden {order_id}...")
            cancel_result = await client.cancel_order(order_id)
            
            if cancel_result:
                print(f"✅ Orden {order_id} cancelada correctamente")
            else:
                print(f"❌ No se pudo cancelar la orden {order_id}")
            
            # 6. Verificar que no hay órdenes
            print(f"\n🔍 Verificando nuevamente órdenes abiertas...")
            await asyncio.sleep(1)
            
            orders = await client.get_orders(status="open")
            if not orders:
                print("✅ No hay órdenes abiertas, perfecto!")
            else:
                print(f"⚠️ Todavía hay {len(orders)} órdenes abiertas")
                
            print("\n🎉 Demostración de integración completada con éxito! 🎉")
            
    except Exception as e:
        print(f"❌ Error durante la demostración: {e}")

def main():
    """Punto de entrada principal."""
    # Ejecutar la demo asincrónica
    asyncio.run(run_integration_demo())

if __name__ == "__main__":
    main()