"""
News Scraper: Módulo para obtener noticias sobre criptomonedas, stocks y ballenas.

Este módulo permite que Buddha AI obtenga noticias relevantes directamente
de internet, para mostrarlas en un carousel en la interfaz y utilizarlas
en el análisis de sentimiento del mercado.
"""

import aiohttp
import asyncio
import logging
import json
import re
import os
import random
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import html

# Configuración de logging
logger = logging.getLogger("genesis.trading.news_scraper")

class NewsItem:
    """Representa una noticia individual con toda su metadata."""
    
    def __init__(
        self,
        title: str,
        url: str,
        source: str,
        date: Optional[datetime] = None,
        summary: str = "",
        image_url: str = "",
        categories: List[str] = None,
        sentiment: float = 0.0,
        relevance: float = 0.0
    ):
        """
        Inicializar noticia.
        
        Args:
            title: Título de la noticia
            url: URL de la noticia
            source: Fuente de la noticia (sitio web)
            date: Fecha de publicación
            summary: Resumen o extracto de la noticia
            image_url: URL de la imagen destacada
            categories: Categorías relevantes (crypto, stock, whale, etc.)
            sentiment: Sentimiento de la noticia (-1.0 a 1.0)
            relevance: Relevancia para el trading (0.0 a 1.0)
        """
        self.title = title
        self.url = url
        self.source = source
        self.date = date or datetime.now()
        self.summary = summary
        self.image_url = image_url
        self.categories = categories or []
        self.sentiment = sentiment
        self.relevance = relevance
        self.processed = False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir noticia a diccionario.
        
        Returns:
            Diccionario con los datos de la noticia
        """
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "date": self.date.isoformat(),
            "summary": self.summary,
            "image_url": self.image_url,
            "categories": self.categories,
            "sentiment": self.sentiment,
            "relevance": self.relevance,
            "processed": self.processed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsItem':
        """
        Crear noticia desde diccionario.
        
        Args:
            data: Diccionario con datos de la noticia
            
        Returns:
            Instancia de noticia
        """
        date = datetime.fromisoformat(data["date"]) if isinstance(data["date"], str) else data["date"]
        return cls(
            title=data["title"],
            url=data["url"],
            source=data["source"],
            date=date,
            summary=data["summary"],
            image_url=data["image_url"],
            categories=data["categories"],
            sentiment=data["sentiment"],
            relevance=data["relevance"]
        )

class NewsScraper:
    """
    Scraper de noticias para obtener información actualizada sobre mercados.
    
    Esta clase permite obtener noticias de diversas fuentes especializadas
    en criptomonedas, mercados financieros y movimientos de ballenas.
    """
    
    def __init__(self, cache_file: str = "cached_news.json"):
        """
        Inicializar scraper de noticias.
        
        Args:
            cache_file: Ruta al archivo de caché para noticias
        """
        self.cache_file = cache_file
        self.sources = {
            "crypto": [
                "cointelegraph.com",
                "coindesk.com",
                "decrypt.co",
                "bitcoinist.com",
                "cryptoslate.com"
            ],
            "finance": [
                "finance.yahoo.com",
                "marketwatch.com",
                "investing.com",
                "bloomberg.com",
                "cnbc.com"
            ],
            "whales": [
                "whalealert.io",
                "whales.io",
                "whalestats.com",
                "whale-alert.io"
            ]
        }
        self.news_cache: List[NewsItem] = []
        self.last_update = datetime.now() - timedelta(days=1)  # Forzar actualización inicial
        self.update_interval = timedelta(hours=1)  # Actualizar cada hora
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Cargar noticias desde caché."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.news_cache = [NewsItem.from_dict(item) for item in data]
                    logger.info(f"Cargadas {len(self.news_cache)} noticias desde caché")
        except Exception as e:
            logger.error(f"Error cargando caché de noticias: {e}")
    
    def _save_cache(self) -> None:
        """Guardar noticias en caché."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump([item.to_dict() for item in self.news_cache], f, ensure_ascii=False, indent=2)
                logger.info(f"Guardadas {len(self.news_cache)} noticias en caché")
        except Exception as e:
            logger.error(f"Error guardando caché de noticias: {e}")
    
    async def get_news(self, category: str = "all", limit: int = 10, force_update: bool = False) -> List[NewsItem]:
        """
        Obtener noticias, actualizando si es necesario.
        
        Args:
            category: Categoría de noticias ('crypto', 'finance', 'whales', 'all')
            limit: Número máximo de noticias a retornar
            force_update: Si es True, fuerza actualización aunque no toque
            
        Returns:
            Lista de noticias ordenadas por fecha (más recientes primero)
        """
        # Verificar si toca actualizar
        now = datetime.now()
        if force_update or (now - self.last_update) > self.update_interval:
            await self.update_news()
        
        # Filtrar por categoría
        if category != "all":
            filtered_news = [
                item for item in self.news_cache 
                if category in item.categories
            ]
        else:
            filtered_news = self.news_cache
        
        # Ordenar por fecha (más recientes primero) y limitar cantidad
        sorted_news = sorted(
            filtered_news, 
            key=lambda x: x.date, 
            reverse=True
        )[:limit]
        
        return sorted_news
    
    async def update_news(self) -> None:
        """Actualizar noticias de todas las categorías."""
        logger.info("Iniciando actualización de noticias...")
        
        try:
            # Obtener noticias nuevas
            new_items = []
            
            for category, sources in self.sources.items():
                for source in sources:
                    try:
                        items = await self._scrape_source(source, category)
                        new_items.extend(items)
                        # Pequeña pausa para no sobrecargar sitios
                        await asyncio.sleep(random.uniform(1.0, 3.0))
                    except Exception as e:
                        logger.error(f"Error scrapeando {source}: {e}")
            
            # Filtrar duplicados (misma URL)
            existing_urls = {item.url for item in self.news_cache}
            unique_new_items = [item for item in new_items if item.url not in existing_urls]
            
            if unique_new_items:
                logger.info(f"Se encontraron {len(unique_new_items)} noticias nuevas")
                
                # Añadir al caché manteniendo un límite
                self.news_cache.extend(unique_new_items)
                
                # Mantener cache en tamaño razonable (máximo 1000 noticias)
                if len(self.news_cache) > 1000:
                    # Ordenar por fecha y mantener las más recientes
                    self.news_cache = sorted(
                        self.news_cache,
                        key=lambda x: x.date,
                        reverse=True
                    )[:1000]
                
                # Guardar en caché
                self._save_cache()
            else:
                logger.info("No se encontraron noticias nuevas")
            
            # Actualizar timestamp
            self.last_update = datetime.now()
        
        except Exception as e:
            logger.error(f"Error actualizando noticias: {e}")
    
    async def _scrape_source(self, source: str, category: str) -> List[NewsItem]:
        """
        Scrapear noticias de una fuente específica.
        
        Args:
            source: URL base de la fuente
            category: Categoría para las noticias
            
        Returns:
            Lista de noticias obtenidas
        """
        try:
            # Determinar URL completa
            url = f"https://{source}"
            
            # Obtener HTML de la página
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }) as response:
                    if response.status != 200:
                        logger.warning(f"Error obteniendo {url}: {response.status}")
                        return []
                    
                    html_content = await response.text()
            
            # Parsear HTML
            items = self._extract_news_from_html(html_content, source, category)
            logger.info(f"Obtenidas {len(items)} noticias de {source}")
            return items
        
        except Exception as e:
            logger.error(f"Error scrapeando {source}: {e}")
            return []
    
    def _extract_news_from_html(self, html_content: str, source: str, category: str) -> List[NewsItem]:
        """
        Extraer noticias del HTML de una página.
        
        Args:
            html_content: Contenido HTML
            source: Nombre de la fuente
            category: Categoría de la fuente
            
        Returns:
            Lista de noticias extraídas
        """
        items = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extraer artículos o noticias (estrategia genérica)
            # Esto funciona en muchos sitios pero puede requerir ajustes específicos
            articles = soup.find_all(['article', 'div'], class_=lambda c: c and ('article' in c.lower() or 'news' in c.lower() or 'post' in c.lower()))
            
            # Si no se encontraron artículos, buscar enlaces con estructura de noticia
            if not articles:
                links = soup.find_all('a', href=True)
                for link in links:
                    # Verificar si el enlace parece una noticia (tiene título, imagen, etc.)
                    title_elem = link.find(['h1', 'h2', 'h3', 'h4', 'h5', 'strong'])
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text().strip()
                    if not title or len(title) < 15:  # Ignorar enlaces con títulos muy cortos
                        continue
                    
                    # Construir URL completa
                    url = link['href']
                    if not url.startswith('http'):
                        url = f"https://{source}{url if url.startswith('/') else '/' + url}"
                    
                    # Extraer imagen si existe
                    image_url = ""
                    img = link.find('img', src=True)
                    if img and img.get('src'):
                        image_src = img['src']
                        image_url = image_src if image_src.startswith('http') else f"https://{source}{image_src if image_src.startswith('/') else '/' + image_src}"
                    
                    # Extracto o resumen (a veces está en un párrafo dentro del enlace)
                    summary = ""
                    p = link.find('p')
                    if p:
                        summary = p.get_text().strip()
                    
                    # Fecha (si está disponible)
                    date = datetime.now()  # Por defecto usar fecha actual
                    date_elem = link.find(['time', 'span'], attrs={'datetime': True})
                    if date_elem and date_elem.get('datetime'):
                        try:
                            date = datetime.fromisoformat(date_elem['datetime'].replace('Z', '+00:00'))
                        except:
                            pass
                    
                    # Crear noticia
                    news_item = NewsItem(
                        title=title,
                        url=url,
                        source=source,
                        date=date,
                        summary=summary,
                        image_url=image_url,
                        categories=[category],
                        sentiment=0.0,  # Se calculará después
                        relevance=0.5   # Valor por defecto
                    )
                    items.append(news_item)
            
            else:
                # Procesar artículos encontrados
                for article in articles:
                    # Extraer título
                    title_elem = article.find(['h1', 'h2', 'h3', 'h4', 'h5'])
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text().strip()
                    if not title:
                        continue
                    
                    # Extraer enlace
                    link = None
                    if title_elem.find('a'):
                        link = title_elem.find('a')
                    elif article.find('a'):
                        link = article.find('a')
                    
                    if not link or not link.get('href'):
                        continue
                    
                    url = link['href']
                    if not url.startswith('http'):
                        url = f"https://{source}{url if url.startswith('/') else '/' + url}"
                    
                    # Extraer imagen
                    image_url = ""
                    img = article.find('img', src=True)
                    if img and img.get('src'):
                        image_src = img['src']
                        image_url = image_src if image_src.startswith('http') else f"https://{source}{image_src if image_src.startswith('/') else '/' + image_src}"
                    
                    # Extraer resumen
                    summary = ""
                    p = article.find('p')
                    if p:
                        summary = p.get_text().strip()
                    
                    # Extraer fecha
                    date = datetime.now()  # Por defecto usar fecha actual
                    date_elem = article.find(['time', 'span'], attrs={'datetime': True})
                    if date_elem and date_elem.get('datetime'):
                        try:
                            date = datetime.fromisoformat(date_elem['datetime'].replace('Z', '+00:00'))
                        except:
                            pass
                    
                    # Crear noticia
                    news_item = NewsItem(
                        title=html.unescape(title),  # Decodificar entidades HTML
                        url=url,
                        source=source,
                        date=date,
                        summary=html.unescape(summary) if summary else "",
                        image_url=image_url,
                        categories=[category],
                        sentiment=0.0,  # Se calculará después
                        relevance=0.5   # Valor por defecto
                    )
                    items.append(news_item)
            
            # Limitar a 10 noticias por fuente para no sobrecargar
            return items[:10]
        
        except Exception as e:
            logger.error(f"Error extrayendo noticias de {source}: {e}")
            return []
    
    async def analyze_sentiment(self, item: NewsItem, buddha_analyzer=None) -> float:
        """
        Analizar sentimiento de una noticia.
        
        Args:
            item: Noticia a analizar
            buddha_analyzer: Analizador de Buddha (opcional)
            
        Returns:
            Puntuación de sentimiento (-1.0 a 1.0)
        """
        # Si ya está calculado, devolver directamente
        if item.processed:
            return item.sentiment
        
        try:
            # Si hay analizador de Buddha, usarlo
            if buddha_analyzer:
                # Preparar texto para análisis
                text = f"{item.title}. {item.summary}"
                
                # Analizar sentimiento con Buddha
                sentiment = await buddha_analyzer.analyze_sentiment(text)
                
                # Actualizar noticia
                item.sentiment = sentiment
                item.processed = True
                
                # Guardar caché actualizado
                self._save_cache()
                
                return sentiment
            
            # Si no hay analizador, usar análisis simple
            return self._simple_sentiment_analysis(item)
        
        except Exception as e:
            logger.error(f"Error analizando sentimiento: {e}")
            return 0.0  # Neutral por defecto
    
    def _simple_sentiment_analysis(self, item: NewsItem) -> float:
        """
        Análisis de sentimiento simple basado en palabras clave.
        
        Args:
            item: Noticia a analizar
            
        Returns:
            Puntuación de sentimiento (-1.0 a 1.0)
        """
        # Diccionarios de palabras positivas y negativas
        positive_words = {
            'subió', 'aumentó', 'creció', 'ganancias', 'beneficios', 'positivo', 'alcista',
            'optimista', 'éxito', 'logro', 'mejora', 'recuperación', 'bullish', 'bull',
            'rally', 'oportunidad', 'auge', 'crecimiento', 'innovación', 'adopción',
            'respaldo', 'éxito', 'respaldo', 'impresionante', 'rompe', 'supera',
            'histórico', 'récord', 'lanzamiento', 'alianza', 'asociación', 'colaboración'
        }
        
        negative_words = {
            'cayó', 'bajó', 'disminuyó', 'pérdidas', 'negativo', 'bajista', 'pesimista',
            'fracaso', 'problema', 'deterioro', 'caída', 'bearish', 'bear', 'corrección',
            'riesgo', 'colapso', 'crisis', 'preocupación', 'amenaza', 'regulación',
            'prohibición', 'fraude', 'hack', 'ataque', 'investigación', 'pánico',
            'incertidumbre', 'desplome', 'miedo', 'advertencia', 'sanción', 'multa'
        }
        
        # Combinar título y resumen
        text = f"{item.title.lower()} {item.summary.lower()}"
        
        # Contar palabras positivas y negativas
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Calcular sentimiento
        total = positive_count + negative_count
        if total == 0:
            sentiment = 0.0  # Neutral
        else:
            sentiment = (positive_count - negative_count) / total
        
        # Actualizar noticia
        item.sentiment = sentiment
        item.processed = True
        
        # Guardar caché actualizado
        self._save_cache()
        
        return sentiment
    
    def get_market_sentiment(self, category: str = "all", timeframe_hours: int = 24) -> float:
        """
        Calcular sentimiento general del mercado basado en noticias recientes.
        
        Args:
            category: Categoría de noticias ('crypto', 'finance', 'whales', 'all')
            timeframe_hours: Periodo de tiempo a considerar en horas
            
        Returns:
            Puntuación de sentimiento general (-1.0 a 1.0)
        """
        # Filtrar noticias por categoría y timeframe
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        # Noticias en el periodo
        if category != "all":
            filtered_news = [
                item for item in self.news_cache 
                if category in item.categories and item.date >= cutoff_time and item.processed
            ]
        else:
            filtered_news = [
                item for item in self.news_cache 
                if item.date >= cutoff_time and item.processed
            ]
        
        # Si no hay noticias procesadas, retornar neutral
        if not filtered_news:
            return 0.0
        
        # Calcular sentimiento promedio ponderado por relevancia
        total_weight = sum(item.relevance for item in filtered_news)
        
        if total_weight == 0:
            # Si no hay pesos, promedio simple
            return sum(item.sentiment for item in filtered_news) / len(filtered_news)
        
        # Promedio ponderado
        weighted_sentiment = sum(item.sentiment * item.relevance for item in filtered_news) / total_weight
        
        return weighted_sentiment