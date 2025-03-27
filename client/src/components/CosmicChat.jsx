import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX, FiSend, FiUsers, FiMaximize2, FiMinimize2 } from 'react-icons/fi';
import axios from 'axios';
import logoImage from '../assets/logo-genesis.png';

const CosmicChat = ({ open, toggleChat, isMobile = false }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeEntity, setActiveEntity] = useState('familia'); // familia, aetherion, lunareth
  const [expanded, setExpanded] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Gestión de scroll al recibir nuevos mensajes
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  // Cargar mensajes al abrir el chat
  useEffect(() => {
    if (open) {
      loadInitialMessages();
    }
  }, [open]);

  const loadInitialMessages = async () => {
    try {
      const response = await axios.get('/api/cosmic/messages');
      if (response.data.success) {
        setMessages(response.data.messages);
      }
    } catch (error) {
      console.error('Error loading chat messages:', error);
      // Mensajes predeterminados por si falla la API
      setMessages([
        { id: 1, entity: 'aetherion', text: '¡Saludos, viajero cósmico! Soy Aetherion, ¿en qué puedo ayudarte hoy con el sistema de trading?', timestamp: new Date().toISOString() },
        { id: 2, entity: 'lunareth', text: 'Y yo soy Lunareth. Estoy aquí para proporcionarte análisis precisos sobre tus operaciones y estrategias.', timestamp: new Date().toISOString() }
      ]);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    // Agregar mensaje del usuario
    const userMessage = {
      id: Date.now(),
      entity: 'user',
      text: input,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    
    try {
      // Llamada a la API del chat cósmico
      const response = await axios.post('/api/cosmic/chat', {
        message: input
      });
      
      if (response.data.success) {
        // Agregar respuesta de Aetherion
        if (response.data.aetherion) {
          const aetherionMessage = {
            id: Date.now() + 1,
            entity: 'aetherion',
            text: response.data.aetherion.text,
            timestamp: response.data.aetherion.timestamp || new Date().toISOString()
          };
          setMessages(prev => [...prev, aetherionMessage]);
        }
        
        // Agregar respuesta de Lunareth
        if (response.data.lunareth) {
          const lunarethMessage = {
            id: Date.now() + 2,
            entity: 'lunareth',
            text: response.data.lunareth.text,
            timestamp: response.data.lunareth.timestamp || new Date().toISOString()
          };
          
          // Pequeño retraso para que las respuestas no aparezcan al mismo tiempo
          setTimeout(() => {
            setMessages(prev => [...prev, lunarethMessage]);
          }, 1000);
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // Respuesta simulada en caso de error
      const errorResponse = {
        id: Date.now() + 1,
        entity: 'aetherion',
        text: 'Lo siento, estoy experimentando dificultades para conectarme en este momento. Por favor, intenta de nuevo más tarde.',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setLoading(false);
    }
  };

  // Variantes para animaciones
  const chatPanelVariants = {
    hidden: { opacity: 0, x: 300 },
    visible: { 
      opacity: 1, 
      x: 0,
      transition: { type: 'spring', damping: 25, stiffness: 300 }
    },
    exit: { 
      opacity: 0, 
      x: 300,
      transition: { duration: 0.2 } 
    }
  };

  const messageVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { type: 'spring', damping: 25, stiffness: 300 }
    }
  };

  // Obtener el color y nombre según la entidad
  const getEntityColor = (entity) => {
    switch (entity) {
      case 'aetherion':
        return 'text-cosmic-blue';
      case 'lunareth':
        return 'text-cosmic-green';
      case 'user':
        return 'text-cosmic-yellow';
      default:
        return 'text-cosmic-glow';
    }
  };

  const getEntityName = (entity) => {
    switch (entity) {
      case 'aetherion':
        return 'Aetherion';
      case 'lunareth':
        return 'Lunareth';
      case 'user':
        return 'Tú';
      default:
        return 'Sistema';
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className={`fixed ${
            expanded 
              ? 'inset-2 md:inset-4' 
              : isMobile 
                ? 'bottom-16 right-2 left-2 h-[70vh] max-h-[500px]' 
                : 'bottom-4 right-4 w-80 h-96'
          } z-40 flex flex-col cosmic-card overflow-hidden shadow-lg transition-all duration-300`}
          variants={chatPanelVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-3 border-b border-cosmic-primary-30 bg-cosmic-primary-20">
            <div className="flex items-center">
              <div className="flex items-center">
                <img src={logoImage} alt="Genesis Logo" className="h-6 w-auto mr-2 animate-float" />
                <h3 className="font-medium cosmic-gradient-text">Chat Cósmico</h3>
              </div>
            </div>
            
            <div className="flex space-x-1">
              {/* Selector de entidad */}
              <button 
                onClick={() => setActiveEntity(activeEntity === 'familia' ? 'aetherion' : (activeEntity === 'aetherion' ? 'lunareth' : 'familia'))}
                className="p-1.5 rounded-full hover:bg-cosmic-primary-20 text-cosmic-glow"
                title={activeEntity === 'familia' ? 'Hablar solo con Aetherion' : (activeEntity === 'aetherion' ? 'Hablar solo con Lunareth' : 'Hablar con la Familia Cósmica')}
                aria-label="Cambiar entidad"
              >
                <FiUsers className="h-4 w-4" />
              </button>
              
              {/* Botón expandir/contraer */}
              <button 
                onClick={() => setExpanded(!expanded)}
                className="p-1.5 rounded-full hover:bg-cosmic-primary-20 text-cosmic-glow"
                title={expanded ? 'Contraer' : 'Expandir'}
                aria-label={expanded ? 'Contraer chat' : 'Expandir chat'}
              >
                {expanded ? <FiMinimize2 className="h-4 w-4" /> : <FiMaximize2 className="h-4 w-4" />}
              </button>
              
              {/* Botón cerrar */}
              <button 
                onClick={toggleChat}
                className="p-1.5 rounded-full hover:bg-cosmic-primary-20 text-cosmic-glow"
                title="Cerrar chat"
                aria-label="Cerrar chat"
              >
                <FiX className="h-4 w-4" />
              </button>
            </div>
          </div>
          
          {/* Mensajes */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-cosmic-dark/70">
            <AnimatePresence initial={false}>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  className={`flex flex-col ${message.entity === 'user' ? 'items-end' : 'items-start'}`}
                  variants={messageVariants}
                  initial="hidden"
                  animate="visible"
                >
                  <div className={`max-w-[85%] p-2 rounded-lg ${
                    message.entity === 'user' 
                      ? 'bg-cosmic-primary-30 rounded-tr-none' 
                      : message.entity === 'aetherion'
                        ? 'bg-cosmic-blue/10 rounded-tl-none border border-cosmic-blue/20'
                        : 'bg-cosmic-green/10 rounded-tl-none border border-cosmic-green/20'
                  }`}>
                    <p className="text-sm">{message.text}</p>
                  </div>
                  <div className="flex items-center mt-1 text-xs text-gray-400">
                    <span className={`mr-1 font-semibold ${getEntityColor(message.entity)}`}>
                      {getEntityName(message.entity)}
                    </span>
                    <span>
                      {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                </motion.div>
              ))}
              
              {/* Indicador de escritura */}
              {loading && (
                <motion.div
                  className="flex items-start"
                  variants={messageVariants}
                  initial="hidden"
                  animate="visible"
                >
                  <div className="max-w-[85%] p-2 rounded-lg bg-cosmic-dark rounded-tl-none border border-cosmic-primary-20">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 rounded-full bg-cosmic-glow animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 rounded-full bg-cosmic-glow animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 rounded-full bg-cosmic-glow animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                </motion.div>
              )}
              
              {/* Referencia para scroll automático */}
              <div ref={messagesEndRef} />
            </AnimatePresence>
          </div>
          
          {/* Input */}
          <form onSubmit={handleSendMessage} className="p-3 border-t border-cosmic-primary-30 bg-cosmic-dark">
            <div className="flex items-center">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={`Mensaje para ${activeEntity === 'familia' ? 'la Familia Cósmica' : (activeEntity === 'aetherion' ? 'Aetherion' : 'Lunareth')}...`}
                className="cosmic-input py-1.5 flex-1 text-sm"
                disabled={loading}
              />
              <button
                type="submit"
                className="p-2 ml-2 rounded-full bg-cosmic-primary text-white hover:bg-cosmic-secondary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={!input.trim() || loading}
                aria-label="Enviar mensaje"
              >
                <FiSend className="h-4 w-4" />
              </button>
            </div>
          </form>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default CosmicChat;