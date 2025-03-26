import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const Index = () => {
  return (
    <div className="index-container">
      <motion.div 
        className="content"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <header>
          <div className="logo-container">
            <motion.div 
              className="logo"
              animate={{ 
                rotate: 360,
                scale: [1, 1.1, 1],
              }}
              transition={{ 
                rotate: { duration: 20, repeat: Infinity, ease: "linear" },
                scale: { duration: 3, repeat: Infinity, ease: "easeInOut" }
              }}
            >
              G
            </motion.div>
          </div>
          <h1>Proyecto Genesis</h1>
          <p className="subtitle">El futuro de las inversiones inteligentes</p>
        </header>

        <section className="history">
          <h2>Historia de Genesis</h2>
          <p>
            El Proyecto Genesis nació como una visión para revolucionar el mundo de las inversiones
            mediante la integración de inteligencia artificial avanzada y análisis predictivo.
            Nuestro sistema combina lo mejor de la tecnología moderna con estrategias de inversión
            probadas para ofrecer resultados excepcionales.
          </p>
          <p>
            Desde su concepción, Genesis ha evolucionado para incluir capacidades de análisis
            de mercado en tiempo real, detección de patrones, y toma de decisiones basada en
            múltiples factores de riesgo y oportunidad.
          </p>
        </section>

        <motion.div 
          className="cta-button"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Link to="/login">Ingresar al Sistema</Link>
        </motion.div>
      </motion.div>

      <div className="cosmic-background">
        {/* Elementos visuales de fondo (estrellas, nebulosas, etc.) */}
      </div>
    </div>
  );
};

export default Index;