"""
Analizador de sentimiento para criptomonedas.

Este módulo proporciona funcionalidades para obtener y analizar
el sentimiento del mercado a partir de fuentes como redes sociales,
noticias y foros de discusión.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import re
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime, timedelta

# Importación condicional para bibliotecas opcionales
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

class SentimentAnalyzer:
    """
    Analizador de sentimiento para criptomonedas.
    
    Proporciona herramientas para analizar el sentimiento del mercado
    basado en datos de redes sociales, noticias y foros.
    """
    
    def __init__(self, 
                 x_api_key: Optional[str] = None,
                 x_api_secret: Optional[str] = None,
                 cache_dir: str = './cache/sentiment',
                 use_mock: bool = False):
        """
        Inicializar analizador de sentimiento.
        
        Args:
            x_api_key: API key para X (Twitter)
            x_api_secret: API secret para X (Twitter)
            cache_dir: Directorio para caché
            use_mock: Si es True, genera datos simulados cuando no hay acceso a API
        """
        self.logger = logging.getLogger(__name__)
        self.x_api_key = x_api_key
        self.x_api_secret = x_api_secret
        self.cache_dir = cache_dir
        self.use_mock = use_mock
        
        # Crear directorio de caché si no existe
        os.makedirs(cache_dir, exist_ok=True)
        
        # Verificar disponibilidad de bibliotecas
        if not NLTK_AVAILABLE and not TEXTBLOB_AVAILABLE:
            self.logger.warning("NLTK y TextBlob no están disponibles. Instale con: pip install nltk textblob")
        
        # Inicializar analizadores
        self.nltk_analyzer = None
        if NLTK_AVAILABLE:
            try:
                self.nltk_analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                self.logger.error(f"Error inicializando NLTK SentimentIntensityAnalyzer: {e}")
        
        self.logger.info("SentimentAnalyzer inicializado")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analizar sentimiento de un texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con puntuaciones de sentimiento
        """
        results = {}
        
        # Analizar con NLTK VADER si está disponible
        if NLTK_AVAILABLE and self.nltk_analyzer is not None:
            try:
                vader_scores = self.nltk_analyzer.polarity_scores(text)
                results['nltk_compound'] = vader_scores['compound']
                results['nltk_positive'] = vader_scores['pos']
                results['nltk_negative'] = vader_scores['neg']
                results['nltk_neutral'] = vader_scores['neu']
            except Exception as e:
                self.logger.error(f"Error en análisis NLTK: {e}")
        
        # Analizar con TextBlob si está disponible
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                results['textblob_polarity'] = blob.sentiment.polarity
                results['textblob_subjectivity'] = blob.sentiment.subjectivity
            except Exception as e:
                self.logger.error(f"Error en análisis TextBlob: {e}")
        
        # Calcular puntuación combinada
        if 'nltk_compound' in results and 'textblob_polarity' in results:
            # Combinar NLTK y TextBlob
            results['combined_score'] = (results['nltk_compound'] + results['textblob_polarity']) / 2
        elif 'nltk_compound' in results:
            results['combined_score'] = results['nltk_compound']
        elif 'textblob_polarity' in results:
            results['combined_score'] = results['textblob_polarity']
        else:
            # Si no hay análisis disponible
            results['combined_score'] = 0.0
        
        return results
    
    async def get_x_posts(self, 
                    query: str, 
                    days_back: int = 1,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener posts de X (Twitter) relacionados con una consulta.
        
        Args:
            query: Consulta de búsqueda
            days_back: Número de días hacia atrás para buscar
            limit: Número máximo de posts a obtener
            
        Returns:
            Lista de posts con metadatos
        """
        if self.x_api_key is None or self.x_api_secret is None:
            if self.use_mock:
                self.logger.warning("API keys de X no configuradas. Usando datos simulados.")
                return self._generate_mock_x_posts(query, days_back, limit)
            else:
                raise ValueError("API keys de X no configuradas. Configure x_api_key y x_api_secret o establezca use_mock=True.")
        
        # Verificar si hay resultados en caché
        cache_file = os.path.join(self.cache_dir, f"x_posts_{query.replace(' ', '_')}_{days_back}d_{limit}.json")
        
        if os.path.exists(cache_file):
            # Verificar si la caché es reciente (menos de 6 horas)
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 6 * 3600:  # 6 horas en segundos
                self.logger.info(f"Usando datos de X en caché para '{query}'")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # Construir parámetros de búsqueda
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Aquí se implementaría la llamada real a la API de X
        # Por ahora, devolvemos datos simulados
        self.logger.warning("Funcionalidad de API de X no implementada. Usando datos simulados.")
        mock_posts = self._generate_mock_x_posts(query, days_back, limit)
        
        # Guardar en caché
        with open(cache_file, 'w') as f:
            json.dump(mock_posts, f)
        
        return mock_posts
    
    def _generate_mock_x_posts(self, 
                             query: str, 
                             days_back: int = 1,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """
        Generar posts simulados de X con sentimiento variado.
        
        Args:
            query: Consulta de búsqueda
            days_back: Número de días hacia atrás
            limit: Número máximo de posts
            
        Returns:
            Lista de posts simulados
        """
        # Generar timestamps aleatorios en el rango especificado
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Lista de símbolos relevantes
        if 'bitcoin' in query.lower() or 'btc' in query.lower():
            symbol = 'BTC'
            crypto_name = 'Bitcoin'
        elif 'ethereum' in query.lower() or 'eth' in query.lower():
            symbol = 'ETH'
            crypto_name = 'Ethereum'
        else:
            symbol = query.upper()
            crypto_name = query.capitalize()
        
        # Templates de posts positivos
        positive_templates = [
            f"{crypto_name} looking bullish today! Target {symbol} 🚀 #crypto",
            f"Just bought more {symbol}! This dip is a gift 💎🙌 #{symbol.lower()}",
            f"The technical analysis for {symbol} is showing strong support. Going up from here! #trading",
            f"{crypto_name} fundamentals are stronger than ever. Hodl gang! 💪",
            f"This {symbol} breakout is just getting started! Next resistance at [price] 📈",
            f"Whales are accumulating {symbol}. Big move coming soon! #crypto",
            f"New partnership for {crypto_name} announced! Bullish news 🔥",
            f"{symbol} volume increasing dramatically. Price action looks good 👍",
            f"The {crypto_name} ecosystem is growing exponentially. Long term bullish! #defi"
        ]
        
        # Templates de posts negativos
        negative_templates = [
            f"{crypto_name} breaking down from key support. Look out below! #{symbol.lower()}",
            f"Selling my {symbol} bag. This downtrend isn't over 📉 #crypto",
            f"Bearish divergence on {symbol} 4h chart. Short opportunity? #trading",
            f"{crypto_name} network activity decreasing. Not a good sign 🚩",
            f"The {symbol} bubble is bursting. Exit while you can! #cryptocrash",
            f"Whale just moved 10,000 {symbol} to exchange. Dump incoming? 😱",
            f"New regulations targeting {crypto_name}. FUD increasing 🧸",
            f"{symbol} looks terrible on all timeframes. Going lower #bear",
            f"I've been a {crypto_name} bull for years but this price action is concerning..."
        ]
        
        # Templates de posts neutrales
        neutral_templates = [
            f"What's everyone's thought on {symbol} price action today? #crypto",
            f"Watching {crypto_name} closely at these levels. Could go either way 🤔",
            f"DYOR on {symbol} before making any decisions. Market is uncertain.",
            f"{crypto_name} consolidating in this range. Waiting for breakout or breakdown.",
            f"New {symbol} update released. Interesting changes! #technology",
            f"Anyone using {crypto_name} for actual transactions? Just curious.",
            f"{symbol} trading volume average today. Nothing unusual to report.",
            f"Comparing {crypto_name} to other L1 blockchains. Thoughts? #blockchain",
            f"Historic volatility for {symbol} decreasing. Range bound for now."
        ]
        
        # Generar posts aleatorios
        mock_posts = []
        
        for i in range(limit):
            # Determinar sentimiento (40% positivo, 30% negativo, 30% neutral)
            sentiment_roll = np.random.random()
            
            if sentiment_roll < 0.4:
                # Post positivo
                text = np.random.choice(positive_templates)
                sentiment = 'positive'
                score = np.random.uniform(0.3, 0.9)
            elif sentiment_roll < 0.7:
                # Post negativo
                text = np.random.choice(negative_templates)
                sentiment = 'negative'
                score = np.random.uniform(-0.9, -0.3)
            else:
                # Post neutral
                text = np.random.choice(neutral_templates)
                sentiment = 'neutral'
                score = np.random.uniform(-0.2, 0.2)
            
            # Personalizar texto para hacerlo más realista
            text = text.replace('[price]', f'${np.random.randint(1000, 100000)}')
            
            # Generar timestamp aleatorio
            post_time = start_time + timedelta(seconds=np.random.randint(0, int((end_time - start_time).total_seconds())))
            
            # Generar datos de engagement
            likes = np.random.randint(0, 1000)
            retweets = np.random.randint(0, 200)
            comments = np.random.randint(0, 50)
            
            # Crear post
            post = {
                'id': f'mock_{i}_{int(time.time())}',
                'text': text,
                'created_at': post_time.isoformat(),
                'user': {
                    'username': f'crypto_user_{np.random.randint(1000, 9999)}',
                    'followers_count': np.random.randint(100, 10000),
                    'verified': np.random.random() < 0.1  # 10% de probabilidad de ser verificado
                },
                'public_metrics': {
                    'like_count': likes,
                    'retweet_count': retweets,
                    'reply_count': comments
                },
                'sentiment': {
                    'label': sentiment,
                    'score': score
                }
            }
            
            mock_posts.append(post)
        
        # Ordenar por timestamp (más recientes primero)
        mock_posts.sort(key=lambda x: x['created_at'], reverse=True)
        
        return mock_posts
    
    async def analyze_social_sentiment(self, 
                                 symbol: str, 
                                 days_back: int = 1,
                                 limit: int = 100,
                                 sources: List[str] = ['x']) -> Dict[str, Any]:
        """
        Analizar sentimiento social para un símbolo.
        
        Args:
            symbol: Símbolo de la criptomoneda
            days_back: Número de días hacia atrás para analizar
            limit: Número máximo de posts a analizar
            sources: Fuentes a usar ('x', 'reddit', etc.)
            
        Returns:
            Diccionario con resultados de sentimiento
        """
        all_posts = []
        sentiment_scores = []
        engagement_weighted_scores = []
        
        # Obtener y analizar posts de X
        if 'x' in sources:
            try:
                # Construir consulta para X
                query = f"{symbol} OR #{symbol.lower()} crypto"
                
                # Obtener posts
                x_posts = await self.get_x_posts(query, days_back, limit)
                
                # Analizar sentimiento para cada post
                for post in x_posts:
                    # Verificar si ya tiene sentimiento (para datos mock)
                    if 'sentiment' in post and 'score' in post['sentiment']:
                        score = post['sentiment']['score']
                    else:
                        # Analizar texto
                        sentiment = self.analyze_text(post['text'])
                        score = sentiment['combined_score']
                        post['sentiment'] = {
                            'score': score,
                            'label': 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
                        }
                    
                    # Calcular peso de engagement
                    engagement = (
                        post['public_metrics']['like_count'] * 1.0 + 
                        post['public_metrics']['retweet_count'] * 2.0 + 
                        post['public_metrics']['reply_count'] * 1.5
                    )
                    
                    # Ajustar peso por número de seguidores
                    followers = post['user']['followers_count']
                    influence_factor = min(1.0, np.log10(followers + 1) / 4)  # Escala logarítmica limitada a 1.0
                    
                    # Bonus por verificación
                    verified_bonus = 1.5 if post.get('user', {}).get('verified', False) else 1.0
                    
                    # Peso final
                    weight = (1.0 + engagement / 1000) * influence_factor * verified_bonus
                    
                    # Guardar score y peso
                    sentiment_scores.append(score)
                    engagement_weighted_scores.append(score * weight)
                
                # Añadir a lista completa
                all_posts.extend(x_posts)
                
            except Exception as e:
                self.logger.error(f"Error al analizar posts de X para {symbol}: {e}")
        
        # Aquí se podrían añadir otras fuentes como Reddit, etc.
        
        # Si no hay posts, devolver resultado vacío
        if not all_posts:
            return {
                'symbol': symbol,
                'sources': sources,
                'days_back': days_back,
                'post_count': 0,
                'sentiment_score': 0.0,
                'weighted_sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'timestamp': datetime.now().isoformat(),
                'posts': []
            }
        
        # Calcular sentimiento global
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # Calcular sentimiento ponderado por engagement
        weighted_sentiment = np.sum(engagement_weighted_scores) / np.sum([np.abs(s) for s in engagement_weighted_scores]) if engagement_weighted_scores else 0.0
        
        # Determinar etiqueta de sentimiento
        if weighted_sentiment > 0.05:
            sentiment_label = 'positive'
        elif weighted_sentiment < -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Calcular distribución de sentimiento
        positive_count = sum(1 for p in all_posts if p.get('sentiment', {}).get('label') == 'positive')
        negative_count = sum(1 for p in all_posts if p.get('sentiment', {}).get('label') == 'negative')
        neutral_count = sum(1 for p in all_posts if p.get('sentiment', {}).get('label') == 'neutral')
        
        total_posts = len(all_posts)
        sentiment_distribution = {
            'positive': positive_count / total_posts if total_posts > 0 else 0,
            'neutral': neutral_count / total_posts if total_posts > 0 else 0,
            'negative': negative_count / total_posts if total_posts > 0 else 0
        }
        
        # Preparar resultado
        result = {
            'symbol': symbol,
            'sources': sources,
            'days_back': days_back,
            'post_count': len(all_posts),
            'sentiment_score': float(avg_sentiment),
            'weighted_sentiment_score': float(weighted_sentiment),
            'sentiment_label': sentiment_label,
            'sentiment_distribution': sentiment_distribution,
            'timestamp': datetime.now().isoformat(),
            'posts': all_posts[:min(10, len(all_posts))]  # Incluir solo los 10 primeros posts para limitar tamaño
        }
        
        # Guardar en caché
        cache_file = os.path.join(self.cache_dir, f"sentiment_{symbol}_{days_back}d.json")
        with open(cache_file, 'w') as f:
            json.dump(result, f)
        
        return result
    
    def plot_sentiment_history(self, 
                              sentiment_data: List[Dict[str, Any]],
                              price_data: Optional[pd.DataFrame] = None,
                              save_path: Optional[str] = None) -> str:
        """
        Generar gráfico de historia de sentimiento.
        
        Args:
            sentiment_data: Lista de resultados de sentimiento
            price_data: DataFrame con datos de precio (opcional)
            save_path: Ruta donde guardar el gráfico
            
        Returns:
            Imagen en formato base64
        """
        if not sentiment_data:
            return ""
        
        # Extraer datos
        timestamps = [datetime.fromisoformat(d['timestamp']) for d in sentiment_data]
        sentiment_scores = [d['weighted_sentiment_score'] for d in sentiment_data]
        
        # Crear figura
        if price_data is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = None
        
        # Graficar sentimiento
        ax1.plot(timestamps, sentiment_scores, 'b-', label='Sentimiento ponderado')
        
        # Añadir línea de sentimiento neutral
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Colorear áreas positivas y negativas
        ax1.fill_between(timestamps, sentiment_scores, 0, where=np.array(sentiment_scores) > 0, color='green', alpha=0.2)
        ax1.fill_between(timestamps, sentiment_scores, 0, where=np.array(sentiment_scores) < 0, color='red', alpha=0.2)
        
        # Formato del gráfico de sentimiento
        symbol = sentiment_data[0]['symbol']
        ax1.set_title(f'Análisis de Sentimiento para {symbol}')
        ax1.set_ylabel('Puntuación de Sentimiento')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graficar precio si está disponible
        if price_data is not None and ax2 is not None:
            # Filtrar datos de precio para el rango de fechas del sentimiento
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