"""
Tests para el adaptador divino de base de datos.

Este módulo contiene pruebas unitarias para verificar la funcionalidad
del adaptador divino de base de datos en diferentes contextos.
"""
import os
import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from genesis.db.divine_database import DivineDatabaseAdapter, DivineCache

# Configuración de pruebas
DB_URL = os.environ.get("DATABASE_URL", "")

# Pruebas para DivineCache
def test_divine_cache():
    """Prueba las funcionalidades básicas del caché divino."""
    # Crear caché con tamaño máximo pequeño para pruebas
    cache = DivineCache(max_size=5, ttl=1)
    
    # Verificar que el caché está vacío inicialmente
    assert cache.get("test_key") is None
    
    # Añadir valores
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", {"complex": "value"})
    
    # Verificar que los valores se almacenan y recuperan correctamente
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3")["complex"] == "value"
    
    # Verificar que el tamaño máximo se respeta
    for i in range(10):
        cache.set(f"overflow_key_{i}", i)
    
    # Debería tener como máximo 5 elementos (max_size)
    stats = cache.get_stats()
    assert stats["size"] <= 5
    
    # Verificar TTL (esperar a que expire)
    cache.set("expire_key", "will_expire", ttl=0.1)
    assert cache.get("expire_key") == "will_expire"
    
    # Esperar a que expire (100ms + margen)
    import time
    time.sleep(0.2)
    assert cache.get("expire_key") is None
    
    # Verificar invalidación por patrón
    cache.set("pattern_key_1", "value1")
    cache.set("pattern_key_2", "value2")
    cache.set("other_key", "value3")
    
    assert cache.invalidate("pattern_key") == 2
    assert cache.get("pattern_key_1") is None
    assert cache.get("pattern_key_2") is None
    assert cache.get("other_key") == "value3"
    
    # Verificar estadísticas
    stats = cache.get_stats()
    assert stats["hits"] > 0
    assert stats["misses"] > 0
    assert stats["sets"] > 0
    assert stats["invalidations"] > 0
    assert "hit_ratio" in stats
    assert "memory_usage_bytes" in stats

class TestDivineDatabaseAdapter:
    """Pruebas para el adaptador divino de base de datos."""
    
    @pytest.fixture
    def db(self):
        """Fixture para obtener una instancia del adaptador."""
        if not DB_URL:
            pytest.skip("No DATABASE_URL environment variable set")
        
        adapter = DivineDatabaseAdapter(DB_URL)
        yield adapter
        adapter.close()
    
    def test_sync_fetch(self, db):
        """Prueba funciones síncronas de fetch."""
        # Contar componentes (debería haber algunos)
        count = db.fetch_val_sync("SELECT count(*) FROM gen_components", default=0)
        assert isinstance(count, int)
        assert count >= 0
        
        # Obtener algunos componentes
        components = db.fetch_all_sync("SELECT * FROM gen_components LIMIT 5")
        
        # Si hay componentes, verificar su estructura
        if components:
            assert isinstance(components, list)
            assert isinstance(components[0], dict)
            assert "id" in components[0] or "component_id" in components[0]
    
    def test_sync_transaction(self, db):
        """Prueba transacciones síncronas."""
        # Solo verificamos que no lance excepciones
        with db.transaction_sync() as tx:
            # Ejecutar una consulta de lectura (no modificamos datos)
            count = tx.fetch_val("SELECT count(*) FROM gen_components")
            assert isinstance(count, int)
    
    def test_cache_behavior(self, db):
        """Prueba comportamiento del caché."""
        # Primera consulta, debe ir a la base de datos
        query = "SELECT count(*) FROM gen_components"
        db.fetch_val_sync(query, use_cache=True)
        
        # Verificar que la segunda consulta use el caché
        stats_before = db.get_stats()
        db.fetch_val_sync(query, use_cache=True)
        stats_after = db.get_stats()
        
        assert stats_after["cache_hits"] > stats_before["cache_hits"]
    
    @pytest.mark.asyncio
    async def test_async_fetch(self, db):
        """Prueba funciones asíncronas de fetch."""
        # Contar componentes (debería haber algunos)
        count = await db.fetch_val_async("SELECT count(*) FROM gen_components", default=0)
        assert isinstance(count, int)
        assert count >= 0
        
        # Obtener algunos componentes
        components = await db.fetch_all_async("SELECT * FROM gen_components LIMIT 5")
        
        # Si hay componentes, verificar su estructura
        if components:
            assert isinstance(components, list)
            assert isinstance(components[0], dict)
            assert "id" in components[0] or "component_id" in components[0]
    
    @pytest.mark.asyncio
    async def test_async_transaction(self, db):
        """Prueba transacciones asíncronas."""
        # Solo verificamos que no lance excepciones
        async with db.transaction_async() as tx:
            # Ejecutar una consulta de lectura (no modificamos datos)
            count = await tx.fetch_val("SELECT count(*) FROM gen_components")
            assert isinstance(count, int)
    
    @pytest.mark.asyncio
    async def test_async_concurrent(self, db):
        """Prueba ejecución concurrente de consultas."""
        # Ejecutar múltiples consultas en paralelo
        tasks = [
            db.fetch_val_async("SELECT count(*) FROM gen_components"),
            db.fetch_val_async("SELECT count(*) FROM gen_intensity_results"),
            db.fetch_val_async("SELECT count(*) FROM gen_processing_cycles")
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verificar que todos los resultados son enteros
        for result in results:
            assert isinstance(result, int)
    
    def test_stats(self, db):
        """Prueba recolección de estadísticas."""
        # Ejecutar algunas consultas para generar estadísticas
        db.fetch_all_sync("SELECT * FROM gen_components LIMIT 5")
        db.fetch_val_sync("SELECT count(*) FROM gen_intensity_results")
        
        # Verificar estadísticas
        stats = db.get_stats()
        assert stats["queries_sync"] >= 2
        assert "query_time_avg" in stats
        assert "cache_stats" in stats

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])