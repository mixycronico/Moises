"""
Script para configurar y verificar las claves de API para el Sistema Genesis.

Este script permite al usuario configurar las claves de API para todas las
integraciones externas, y verificar su estado actual.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('setup_api_keys')

# Importar gestor de APIs
try:
    from genesis.api_integration import api_manager, initialize
except ImportError:
    logger.error("No se pudo importar el módulo genesis.api_integration")
    logger.error("Asegúrate de ejecutar este script desde el directorio raíz del proyecto")
    sys.exit(1)

# Lista de APIs soportadas y sus variables de entorno correspondientes
API_ENV_VARS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
    "news_api": "NEWS_API_KEY",
    "coinmarketcap": "COINMARKETCAP_API_KEY",
    "reddit_client": "REDDIT_CLIENT_ID",
    "reddit_secret": "REDDIT_CLIENT_SECRET"
}

# Descripciones de cada API
API_DESCRIPTIONS = {
    "deepseek": "DeepSeek - API de IA avanzada para análisis de mercado y trading",
    "alpha_vantage": "Alpha Vantage - Datos históricos y fundamentales de mercado",
    "news_api": "NewsAPI - Noticias y análisis de sentimiento",
    "coinmarketcap": "CoinMarketCap - Datos detallados de criptomonedas",
    "reddit_client": "Reddit (Client ID) - Análisis de sentimiento social",
    "reddit_secret": "Reddit (Client Secret) - Análisis de sentimiento social"
}

# Enlaces para obtener las claves de API
API_SIGNUP_LINKS = {
    "deepseek": "https://platform.deepseek.com/",
    "alpha_vantage": "https://www.alphavantage.co/support/#api-key",
    "news_api": "https://newsapi.org/register",
    "coinmarketcap": "https://pro.coinmarketcap.com/signup",
    "reddit": "https://www.reddit.com/prefs/apps"
}

def get_current_api_keys() -> Dict[str, str]:
    """
    Obtener claves API actuales desde variables de entorno.
    
    Returns:
        Diccionario con las claves disponibles
    """
    keys = {}
    for api_name, env_var in API_ENV_VARS.items():
        keys[api_name] = os.environ.get(env_var, "")
    return keys

def mask_api_key(key: str) -> str:
    """
    Enmascarar clave API para mostrarla sin revelar toda la información.
    
    Args:
        key: Clave API completa
        
    Returns:
        Clave enmascarada
    """
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "..." + key[-4:]

def show_current_keys():
    """Mostrar claves API actuales."""
    keys = get_current_api_keys()
    
    print("\n=== CLAVES API CONFIGURADAS ===\n")
    for api_name, key in keys.items():
        description = API_DESCRIPTIONS.get(api_name, api_name)
        status = "CONFIGURADA" if key else "NO CONFIGURADA"
        masked_key = mask_api_key(key) if key else "N/A"
        
        print(f"{description}:")
        print(f"  Estado: {status}")
        if key:
            print(f"  Clave: {masked_key}")
        print()
    
    print("===================================\n")

async def verify_api_keys():
    """Verificar validez de las claves API configuradas."""
    print("\n=== VERIFICANDO CLAVES API ===\n")
    
    # Inicializar gestor de APIs
    await initialize()
    
    # Obtener estado
    status = api_manager.get_api_status()
    
    for api_name, api_status in status.items():
        available = api_status.get("available", False)
        print(f"{api_name.upper()}:")
        print(f"  Disponible: {'Sí' if available else 'No'}")
        
        # Mostrar información adicional específica por API
        if api_name == "reddit":
            client_id = api_status.get("client_id_configured", False)
            client_secret = api_status.get("client_secret_configured", False)
            print(f"  Client ID configurado: {'Sí' if client_id else 'No'}")
            print(f"  Client Secret configurado: {'Sí' if client_secret else 'No'}")
        else:
            key_configured = api_status.get("key_configured", False)
            print(f"  Clave configurada: {'Sí' if key_configured else 'No'}")
            
        print()
    
    print("=============================\n")

def print_setup_instructions():
    """Mostrar instrucciones para configurar claves API."""
    print("\n=== INSTRUCCIONES DE CONFIGURACIÓN ===\n")
    
    print("Para configurar las claves API, sigue estos pasos:\n")
    
    for api_name, link in API_SIGNUP_LINKS.items():
        if api_name == "reddit":
            description = "Reddit (Client ID y Secret)"
        else:
            description = API_DESCRIPTIONS.get(api_name, api_name)
            
        print(f"1. {description}:")
        print(f"   - Regístrate en: {link}")
        print(f"   - Copia la clave API proporcionada")
        if api_name == "reddit":
            print(f"   - Configura en Replit como las variables REDDIT_CLIENT_ID y REDDIT_CLIENT_SECRET")
        else:
            env_var = API_ENV_VARS.get(api_name, f"{api_name.upper()}_API_KEY")
            print(f"   - Configura en Replit como la variable {env_var}")
        print()
    
    print("Para configurar variables de entorno en Replit:")
    print("1. Ve a 'Tools' -> 'Secrets' en el panel lateral")
    print("2. Añade cada clave API como un nuevo secreto")
    print("3. Reinicia tu Repl para que los cambios surtan efecto\n")
    
    print("=================================\n")

async def test_apis():
    """Probar APIs configuradas."""
    print("\n=== PROBANDO APIS CONFIGURADAS ===\n")
    
    # Inicializar gestor de APIs si no se ha hecho antes
    if not hasattr(api_manager, 'initialized') or not api_manager.initialized:
        await initialize()
    
    # Probar cada API
    for api_name in ["alpha_vantage", "news_api", "coinmarketcap", "reddit", "deepseek"]:
        print(f"Probando {api_name.upper()}...")
        
        if not api_manager.is_api_available(api_name):
            print(f"  API no disponible. Clave no configurada o inválida.\n")
            continue
        
        try:
            if api_name == "alpha_vantage":
                result = await api_manager.get_alpha_vantage_data(
                    function="GLOBAL_QUOTE",
                    symbol="BTC"
                )
            elif api_name == "news_api":
                result = await api_manager.get_news_api_data(
                    query="bitcoin",
                    language="es",
                    sortBy="publishedAt",
                    pageSize=1
                )
            elif api_name == "coinmarketcap":
                result = await api_manager.get_coinmarketcap_data(
                    endpoint_path="cryptocurrency/listings/latest",
                    start=1,
                    limit=1,
                    convert="USD"
                )
            elif api_name == "reddit":
                result = await api_manager.get_reddit_data(
                    subreddit="Bitcoin",
                    limit=1
                )
            elif api_name == "deepseek":
                result = await api_manager.call_deepseek_api(
                    messages=[
                        {"role": "system", "content": "Eres un asistente experto en análisis de criptomonedas."},
                        {"role": "user", "content": "Resume en una frase el estado actual de Bitcoin."}
                    ],
                    max_tokens=50
                )
            
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Respuesta exitosa. API funcionando correctamente.")
                
        except Exception as e:
            print(f"  Error al probar la API: {str(e)}")
            
        print()
    
    print("==============================\n")

async def main():
    """Función principal."""
    print("\n=== CONFIGURACIÓN DE APIS PARA SISTEMA GENESIS ===\n")
    
    while True:
        print("Selecciona una opción:")
        print("1. Ver claves API configuradas")
        print("2. Verificar validez de claves API")
        print("3. Probar APIs configuradas")
        print("4. Ver instrucciones de configuración")
        print("q. Salir")
        
        option = input("\nOpción: ").strip().lower()
        
        if option == "1":
            show_current_keys()
        elif option == "2":
            await verify_api_keys()
        elif option == "3":
            await test_apis()
        elif option == "4":
            print_setup_instructions()
        elif option == "q":
            print("\nSaliendo del programa...\n")
            break
        else:
            print("\nOpción no válida. Inténtalo de nuevo.\n")
    
    # Cerrar conexiones
    if hasattr(api_manager, 'session') and api_manager.session:
        await api_manager.close()

if __name__ == "__main__":
    asyncio.run(main())