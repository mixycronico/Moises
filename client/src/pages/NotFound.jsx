import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FiArrowLeft } from 'react-icons/fi';

const NotFound = () => {
  const containerVariants = {
    initial: { opacity: 0 },
    animate: { 
      opacity: 1,
      transition: { duration: 0.5 }
    }
  };
  
  const textVariants = {
    initial: { y: 20, opacity: 0 },
    animate: { 
      y: 0, 
      opacity: 1,
      transition: { delay: 0.2, duration: 0.5 }
    }
  };
  
  const errorCodeVariants = {
    initial: { scale: 0.8, opacity: 0 },
    animate: { 
      scale: 1, 
      opacity: 1,
      transition: { 
        delay: 0.3, 
        duration: 0.6,
        type: "spring",
        stiffness: 100
      }
    }
  };

  return (
    <motion.div 
      className="min-h-screen bg-cosmic-gradient flex flex-col items-center justify-center p-4 text-center"
      variants={containerVariants}
      initial="initial"
      animate="animate"
    >
      <motion.div
        className="text-9xl font-display font-bold cosmic-gradient-text"
        variants={errorCodeVariants}
      >
        404
      </motion.div>
      
      <motion.div
        className="mt-6 mb-8 max-w-md"
        variants={textVariants}
      >
        <h1 className="text-2xl font-semibold mb-2">
          Página no encontrada
        </h1>
        <p className="text-gray-400">
          La página que estás buscando parece haberse perdido en el cosmos. 
          Puede que haya sido absorbida por un agujero negro o simplemente no exista.
        </p>
      </motion.div>
      
      <motion.div
        variants={textVariants}
      >
        <Link 
          to="/"
          className="cosmic-button flex items-center justify-center"
        >
          <FiArrowLeft className="mr-2" />
          Volver al inicio
        </Link>
      </motion.div>
      
      {/* Elementos decorativos flotando */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-2 h-2 rounded-full bg-cosmic-blue opacity-70 animate-ping" style={{ animationDuration: '3s', animationDelay: '0.5s' }}></div>
        <div className="absolute top-1/3 right-1/4 w-3 h-3 rounded-full bg-cosmic-green opacity-60 animate-ping" style={{ animationDuration: '4s', animationDelay: '1s' }}></div>
        <div className="absolute bottom-1/4 left-1/3 w-2 h-2 rounded-full bg-cosmic-highlight opacity-50 animate-ping" style={{ animationDuration: '5s', animationDelay: '1.5s' }}></div>
        <div className="absolute top-2/3 right-1/3 w-1 h-1 rounded-full bg-cosmic-yellow opacity-70 animate-ping" style={{ animationDuration: '2.5s' }}></div>
      </div>
    </motion.div>
  );
};

export default NotFound;