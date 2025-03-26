# analysis.py - Archivo creado automáticamente
import logging

def analyze_logs():
    """Analiza los logs del sistema en busca de patrones sospechosos."""
    logger = logging.getLogger("AdminAnalysis")
    try:
        with open("logs/system.log", "r") as log_file:
            logs = log_file.readlines()
            for line in logs[-50:]:  # Analiza las últimas 50 líneas
                if "ERROR" in line or "🚨" in line:
                    logger.warning(f"🔍 Análisis detectó una anomalía: {line.strip()}")
    except FileNotFoundError:
        logger.error("❌ No se encontró el archivo de logs.")

if __name__ == "__main__":
    analyze_logs()