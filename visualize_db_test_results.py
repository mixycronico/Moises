"""
Script para visualizar los resultados de las pruebas de intensidad de la base de datos.
"""
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_results(file_path):
    """Cargar resultados desde archivo JSON."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error al cargar {file_path}: {e}")
        return None

def visualize_single_intensity(file_path):
    """Visualizar resultados de una sola intensidad."""
    results = load_results(file_path)
    if not results:
        return

    intensity = results.get('intensity', 'UNKNOWN')
    summary = results.get('summary', {})
    
    print(f"\n=== RESULTADOS DE INTENSIDAD {intensity} ===")
    print(f"Fecha: {datetime.fromtimestamp(results.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tasa de éxito: {summary.get('success_rate', 0) * 100:.2f}%")
    print(f"Pruebas exitosas: {summary.get('success_count', 0)}/{summary.get('total_tests', 0)}")
    print(f"Duración total: {summary.get('total_duration', 0):.2f}s\n")
    
    # Extraer y mostrar métricas agregadas
    metrics = summary.get('metrics', {})
    if metrics:
        print("Métricas clave:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  - {key}: {value}")
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                print(f"  - {key}: {np.mean(value):.2f} (promedio de {len(value)} valores)")

def visualize_all_results(file_path):
    """Visualizar resultados de todas las intensidades."""
    all_results = load_results(file_path)
    if not all_results:
        return
    
    print("\n=== RESUMEN DE TODAS LAS PRUEBAS DE INTENSIDAD ===")
    print(f"Fecha: {datetime.fromtimestamp(all_results.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duración total: {all_results.get('duration', 0):.2f}s")
    print(f"Intensidades probadas: {', '.join(all_results.get('intensities_tested', []))}\n")
    
    # Extraer datos para gráficas
    intensities = []
    success_rates = []
    durations = []
    operations = []
    
    for intensity, data in all_results.get('results', {}).items():
        intensities.append(intensity)
        success_rates.append(data.get('success_rate', 0) * 100)
        durations.append(data.get('total_duration', 0))
        
        # Extraer operaciones totales de métricas si están disponibles
        total_ops = 0
        for test in data.get('metrics', {}).get('total_operations_avg', []):
            if isinstance(test, (int, float)):
                total_ops += test
        operations.append(total_ops)
    
    # Crear gráficas
    plt.figure(figsize=(15, 10))
    
    # Gráfica 1: Tasas de éxito
    plt.subplot(2, 2, 1)
    plt.bar(intensities, success_rates, color='green')
    plt.title('Tasa de Éxito por Intensidad')
    plt.ylabel('Tasa de Éxito (%)')
    plt.ylim(0, 105)
    for i, v in enumerate(success_rates):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    # Gráfica 2: Duración total
    plt.subplot(2, 2, 2)
    plt.bar(intensities, durations, color='blue')
    plt.title('Duración Total por Intensidad')
    plt.ylabel('Tiempo (s)')
    for i, v in enumerate(durations):
        plt.text(i, v + 1, f"{v:.1f}s", ha='center')
    
    # Gráfica 3: Operaciones totales
    if operations:
        plt.subplot(2, 2, 3)
        plt.bar(intensities, operations, color='orange')
        plt.title('Operaciones Totales por Intensidad')
        plt.ylabel('Número de Operaciones')
        for i, v in enumerate(operations):
            plt.text(i, v + max(operations)*0.05, f"{int(v)}", ha='center')
    
    # Gráfica 4: Relación operaciones/segundo
    if operations and durations:
        ops_per_second = [o/d if d > 0 else 0 for o, d in zip(operations, durations)]
        plt.subplot(2, 2, 4)
        plt.bar(intensities, ops_per_second, color='purple')
        plt.title('Operaciones por Segundo')
        plt.ylabel('Ops/s')
        for i, v in enumerate(ops_per_second):
            plt.text(i, v + max(ops_per_second)*0.05, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    
    # Guardar gráfica
    output_file = "db_test_results_summary.png"
    plt.savefig(output_file)
    print(f"Gráfica guardada en {output_file}")
    
    # Mostrar tablas de datos
    print("\nResultados por intensidad:")
    print(f"{'Intensidad':<10} {'Tasa de Éxito':<15} {'Duración':<15} {'Operaciones':<15} {'Ops/s':<10}")
    print("-" * 65)
    
    for i in range(len(intensities)):
        ops_per_sec = operations[i]/durations[i] if durations[i] > 0 else 0
        print(f"{intensities[i]:<10} {success_rates[i]:.2f}%{'':<8} {durations[i]:.2f}s{'':<8} {int(operations[i]):<15} {ops_per_sec:.2f}")

def main():
    """Función principal."""
    # Comprobar archivos disponibles
    result_files = [f for f in os.listdir('.') if f.startswith('test_db_intensity_') and f.endswith('_results.json')]
    
    if not result_files:
        print("No se encontraron archivos de resultados. Ejecute primero las pruebas.")
        return
    
    print(f"Archivos de resultados disponibles: {len(result_files)}")
    
    # Si existe el archivo de resultados completo, visualizarlo
    if 'test_db_intensity_all_results.json' in result_files:
        visualize_all_results('test_db_intensity_all_results.json')
    
    # Visualizar resultados individuales
    for file in sorted(result_files):
        if file != 'test_db_intensity_all_results.json':
            visualize_single_intensity(file)

if __name__ == "__main__":
    main()