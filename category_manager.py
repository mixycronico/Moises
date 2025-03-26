"""
Sistema de categorización dinámica para inversionistas de Proto Genesis.

Este módulo implementa la lógica para que Aetherion evalúe y actualice
automáticamente las categorías de los inversionistas según su desempeño,
capital, antigüedad y comportamiento.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
import math
import numpy as np

from models import db, Investor, Transaction

# Configurar logging
logger = logging.getLogger('genesis_website')

# Categorías disponibles en orden de menor a mayor
CATEGORIES = ['bronze', 'silver', 'gold', 'platinum']

# Umbrales de capital para categorías (en USD)
CAPITAL_THRESHOLDS = {
    'bronze': 1000,       # Mínimo $1,000
    'silver': 10000,      # Mínimo $10,000
    'gold': 50000,        # Mínimo $50,000
    'platinum': 100000    # Mínimo $100,000
}

# Umbrales de antigüedad para categorías (en días)
TENURE_THRESHOLDS = {
    'bronze': 0,          # Sin requisito de antigüedad
    'silver': 60,         # Mínimo 2 meses
    'gold': 180,          # Mínimo 6 meses
    'platinum': 365       # Mínimo 1 año
}

# Umbrales de rendimiento para categorías (ROI anualizado %)
PERFORMANCE_THRESHOLDS = {
    'bronze': 0,          # Sin requisito de rendimiento
    'silver': 5,          # Mínimo 5% anual
    'gold': 10,           # Mínimo 10% anual
    'platinum': 15        # Mínimo 15% anual
}

# Períodos para evaluación de recategorización (en días)
EVALUATION_PERIODS = {
    'upgrade': 30,        # Evaluar actualización cada 30 días
    'downgrade': 60       # Evaluar descenso cada 60 días
}

# Factor de ponderación para cada criterio
WEIGHTS = {
    'capital': 0.5,       # El capital tiene un peso del 50%
    'tenure': 0.2,        # La antigüedad tiene un peso del 20%
    'performance': 0.2,   # El rendimiento tiene un peso del 20%
    'behavior': 0.1       # El comportamiento tiene un peso del 10%
}

class InvestorEvaluator:
    """Evaluador de inversionistas para categorización dinámica."""
    
    def __init__(self, investor_id: int):
        """
        Inicializar evaluador para un inversionista específico.
        
        Args:
            investor_id: ID del inversionista a evaluar
        """
        self.investor_id = investor_id
        self.investor = None
        self.stats = {}
        self.current_category = 'bronze'
        self.recommended_category = 'bronze'
        self.evaluation_result = {}
        
    def load_investor_data(self) -> bool:
        """
        Cargar datos del inversionista desde la base de datos.
        
        Returns:
            True si se cargaron correctamente, False en caso contrario
        """
        try:
            self.investor = Investor.query.get(self.investor_id)
            if not self.investor:
                logger.error(f"Inversionista no encontrado: {self.investor_id}")
                return False
            
            self.current_category = self.investor.category.lower()
            return True
        except Exception as e:
            logger.error(f"Error al cargar datos del inversionista: {str(e)}")
            return False
    
    def calculate_stats(self) -> bool:
        """
        Calcular estadísticas del inversionista para evaluación.
        
        Returns:
            True si se calcularon correctamente, False en caso contrario
        """
        try:
            # Calcular antigüedad en días
            tenure_days = (datetime.utcnow() - self.investor.created_at).days
            
            # Calcular rendimiento anualizado
            if self.investor.capital > 0:
                total_return = self.investor.earnings / self.investor.capital
                days_since_creation = max(1, tenure_days)  # Evitar división por cero
                annualized_return = (1 + total_return) ** (365 / days_since_creation) - 1
                roi_percent = annualized_return * 100
            else:
                roi_percent = 0
            
            # Obtener transacciones recientes para análisis de comportamiento
            recent_transactions = Transaction.query.filter_by(
                investor_id=self.investor_id
            ).order_by(Transaction.timestamp.desc()).limit(50).all()
            
            # Calcular puntaje de comportamiento (0-100)
            behavior_score = self._calculate_behavior_score(recent_transactions)
            
            # Guardar estadísticas
            self.stats = {
                'capital': self.investor.capital,
                'tenure_days': tenure_days,
                'roi_percent': roi_percent,
                'behavior_score': behavior_score,
                'last_category_change': self.investor.last_category_change if hasattr(self.investor, 'last_category_change') else None
            }
            
            return True
        except Exception as e:
            logger.error(f"Error al calcular estadísticas: {str(e)}")
            return False
    
    def _calculate_behavior_score(self, transactions: List[Transaction]) -> float:
        """
        Calcular puntaje de comportamiento del inversionista (0-100).
        
        Args:
            transactions: Lista de transacciones recientes
            
        Returns:
            Puntaje de comportamiento (0-100)
        """
        # Iniciar con un puntaje base de 80
        score = 80.0
        
        if not transactions:
            return score
        
        # Analizar patrones de comportamiento
        deposit_count = 0
        withdrawal_count = 0
        consistent_deposits = 0
        
        last_deposit_time = None
        
        for tx in transactions:
            if tx.type == 'deposit':
                deposit_count += 1
                
                # Verificar consistencia en depósitos
                if last_deposit_time:
                    days_between = (tx.timestamp - last_deposit_time).days
                    if 25 <= days_between <= 35:  # Depósito mensual consistente
                        consistent_deposits += 1
                
                last_deposit_time = tx.timestamp
                
            elif tx.type == 'withdrawal':
                withdrawal_count += 1
        
        # Premiar por consistencia en depósitos
        if deposit_count > 0:
            consistency_ratio = consistent_deposits / deposit_count
            score += consistency_ratio * 10
        
        # Penalizar por exceso de retiros
        if deposit_count > 0:
            withdrawal_ratio = withdrawal_count / deposit_count
            if withdrawal_ratio > 0.8:  # Más de 80% de depósitos son retirados
                score -= 15
        
        # Limitar puntaje entre 0 y 100
        return max(0, min(100, score))
    
    def evaluate_category(self) -> Dict[str, Any]:
        """
        Evaluar categoría recomendada basada en estadísticas del inversionista.
        
        Returns:
            Diccionario con resultado de evaluación
        """
        try:
            # Verificar si se han cargado los datos y estadísticas
            if not self.investor or not self.stats:
                raise ValueError("Datos o estadísticas no disponibles")
            
            # Calcular puntajes para cada categoría
            category_scores = {}
            
            for category in CATEGORIES:
                # Puntaje por capital (0-1)
                capital_threshold = CAPITAL_THRESHOLDS[category]
                capital_score = min(1.0, self.stats['capital'] / capital_threshold)
                
                # Puntaje por antigüedad (0-1)
                tenure_threshold = TENURE_THRESHOLDS[category]
                tenure_score = min(1.0, self.stats['tenure_days'] / max(1, tenure_threshold))
                
                # Puntaje por rendimiento (0-1)
                performance_threshold = PERFORMANCE_THRESHOLDS[category]
                performance_score = min(1.0, self.stats['roi_percent'] / max(0.1, performance_threshold))
                
                # Puntaje por comportamiento (0-1)
                behavior_score = self.stats['behavior_score'] / 100
                
                # Calcular puntaje ponderado total
                weighted_score = (
                    WEIGHTS['capital'] * capital_score +
                    WEIGHTS['tenure'] * tenure_score +
                    WEIGHTS['performance'] * performance_score +
                    WEIGHTS['behavior'] * behavior_score
                )
                
                category_scores[category] = {
                    'total_score': weighted_score,
                    'capital_score': capital_score,
                    'tenure_score': tenure_score,
                    'performance_score': performance_score,
                    'behavior_score': behavior_score
                }
            
            # Determinar categoría recomendada
            eligible_categories = []
            
            for category in CATEGORIES:
                # Verificar cumplimiento de requisitos mínimos
                meets_capital = self.stats['capital'] >= CAPITAL_THRESHOLDS[category]
                meets_tenure = self.stats['tenure_days'] >= TENURE_THRESHOLDS[category]
                meets_performance = self.stats['roi_percent'] >= PERFORMANCE_THRESHOLDS[category]
                
                # Una categoría es elegible si cumple al menos con capital y antigüedad
                if meets_capital and meets_tenure:
                    eligible_categories.append(category)
            
            # Si no hay categorías elegibles, asignar Bronze
            if not eligible_categories:
                recommended_category = 'bronze'
            else:
                # Obtener la categoría elegible más alta
                recommended_category = eligible_categories[-1]
            
            # Guardar la categoría recomendada
            self.recommended_category = recommended_category
            
            # Verificar si la categoria actual es la maxima?
            current_idx = CATEGORIES.index(self.current_category)
            recommended_idx = CATEGORIES.index(self.recommended_category)
            
            # Determinar si es subida o bajada de categoría
            if current_idx < recommended_idx:
                change_type = 'upgrade'
            elif current_idx > recommended_idx:
                change_type = 'downgrade'
            else:
                change_type = 'maintain'
            
            # Revisar si se cumple el período desde el último cambio de categoría
            should_apply_change = True
            
            if self.stats.get('last_category_change'):
                days_since_last_change = (datetime.utcnow() - self.stats['last_category_change']).days
                
                if change_type == 'upgrade' and days_since_last_change < EVALUATION_PERIODS['upgrade']:
                    should_apply_change = False
                elif change_type == 'downgrade' and days_since_last_change < EVALUATION_PERIODS['downgrade']:
                    should_apply_change = False
            
            # Construir resultado
            self.evaluation_result = {
                'investor_id': self.investor_id,
                'current_category': self.current_category,
                'recommended_category': self.recommended_category,
                'change_type': change_type,
                'should_apply_change': should_apply_change,
                'stats': self.stats,
                'category_scores': category_scores,
                'evaluation_date': datetime.utcnow()
            }
            
            return self.evaluation_result
        
        except Exception as e:
            logger.error(f"Error al evaluar categoría: {str(e)}")
            return {
                'error': str(e),
                'investor_id': self.investor_id,
                'current_category': self.current_category
            }
    
    def apply_category_change(self) -> Dict[str, Any]:
        """
        Aplicar cambio de categoría si es necesario.
        
        Returns:
            Diccionario con resultado de la operación
        """
        try:
            # Verificar que haya evaluación previa
            if not self.evaluation_result:
                raise ValueError("No hay evaluación previa disponible")
            
            # Verificar si se debe aplicar cambio
            if not self.evaluation_result.get('should_apply_change', False):
                return {
                    'success': True,
                    'message': "No se requiere cambio de categoría en este momento",
                    'applied_change': False,
                    'current_category': self.current_category
                }
            
            # Verificar si hay cambio de categoría
            if self.current_category == self.recommended_category:
                return {
                    'success': True,
                    'message': "La categoría actual ya es la recomendada",
                    'applied_change': False,
                    'current_category': self.current_category
                }
            
            # Aplicar cambio de categoría
            old_category = self.current_category
            new_category = self.recommended_category
            
            # Actualizar categoría
            self.investor.category = new_category
            self.investor.last_category_change = datetime.utcnow()
            
            # Guardar en la base de datos
            db.session.commit()
            
            # Registrar cambio
            logger.info(f"Categoría actualizada: Inversionista={self.investor_id}, {old_category} -> {new_category}")
            
            return {
                'success': True,
                'message': f"Categoría actualizada de {old_category} a {new_category}",
                'applied_change': True,
                'old_category': old_category,
                'new_category': new_category,
                'change_date': datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error al aplicar cambio de categoría: {str(e)}")
            return {
                'success': False,
                'message': f"Error al aplicar cambio de categoría: {str(e)}",
                'applied_change': False
            }

def evaluate_investor_category(investor_id: int) -> Dict[str, Any]:
    """
    Evaluar y obtener categoría recomendada para un inversionista.
    
    Args:
        investor_id: ID del inversionista
        
    Returns:
        Resultado de evaluación
    """
    evaluator = InvestorEvaluator(investor_id)
    
    if not evaluator.load_investor_data():
        return {'error': "No se pudo cargar datos del inversionista"}
    
    if not evaluator.calculate_stats():
        return {'error': "No se pudo calcular estadísticas"}
    
    return evaluator.evaluate_category()

def apply_category_change(investor_id: int) -> Dict[str, Any]:
    """
    Evaluar y aplicar cambio de categoría para un inversionista si corresponde.
    
    Args:
        investor_id: ID del inversionista
        
    Returns:
        Resultado de la operación
    """
    evaluator = InvestorEvaluator(investor_id)
    
    if not evaluator.load_investor_data():
        return {'error': "No se pudo cargar datos del inversionista"}
    
    if not evaluator.calculate_stats():
        return {'error': "No se pudo calcular estadísticas"}
    
    evaluator.evaluate_category()
    return evaluator.apply_category_change()

def run_category_evaluation_batch(batch_size: int = 100) -> Dict[str, Any]:
    """
    Ejecutar evaluación y actualización de categorías para un lote de inversionistas.
    Esta función debe ejecutarse periódicamente mediante un trabajo programado.
    
    Args:
        batch_size: Tamaño del lote a procesar
        
    Returns:
        Resultados del procesamiento
    """
    results = {
        'investors_processed': 0,
        'categories_changed': 0,
        'upgrades': 0,
        'downgrades': 0,
        'errors': 0,
        'details': []
    }
    
    try:
        # Obtener inversionistas a evaluar
        # Priorizar aquellos que no han sido evaluados recientemente
        investors = Investor.query.order_by(
            Investor.last_category_evaluation.asc()
        ).limit(batch_size).all()
        
        for investor in investors:
            try:
                # Evaluar y aplicar cambio si corresponde
                evaluator = InvestorEvaluator(investor.id)
                
                if not evaluator.load_investor_data() or not evaluator.calculate_stats():
                    results['errors'] += 1
                    continue
                
                evaluation = evaluator.evaluate_category()
                change_result = evaluator.apply_category_change()
                
                # Actualizar fecha de última evaluación
                investor.last_category_evaluation = datetime.utcnow()
                db.session.commit()
                
                # Registrar resultados
                results['investors_processed'] += 1
                
                if change_result.get('applied_change', False):
                    results['categories_changed'] += 1
                    
                    if evaluation['change_type'] == 'upgrade':
                        results['upgrades'] += 1
                    elif evaluation['change_type'] == 'downgrade':
                        results['downgrades'] += 1
                
                results['details'].append({
                    'investor_id': investor.id,
                    'old_category': evaluation['current_category'],
                    'new_category': evaluation['recommended_category'],
                    'applied': change_result.get('applied_change', False)
                })
                
            except Exception as e:
                results['errors'] += 1
                logger.error(f"Error al procesar inversionista {investor.id}: {str(e)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error en proceso de evaluación por lotes: {str(e)}")
        return {
            'error': str(e),
            'investors_processed': results['investors_processed'],
            'categories_changed': results['categories_changed']
        }