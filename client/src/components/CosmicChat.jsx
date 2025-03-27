import { useState, useEffect, useRef } from 'react';
import { FiSend, FiMinimize2, FiMaximize2, FiX } from 'react-icons/fi';
import gsap from 'gsap';

const CosmicChat = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);

  // Conectar al WebSocket al abrir el chat
  useEffect(() => {
    if (isOpen && !wsRef.current) {
      // Intentar conectar al WebSocket
      console.log('Conectando al WebSocket...');
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;
      
      try {
        wsRef.current = new WebSocket(wsUrl);
        
        wsRef.current.onopen = () => {
          console.log('WebSocket conectado');
          setIsConnected(true);
          
          // Animar entrada de mensajes iniciales
          gsap.fromTo(
            '.message',
            { opacity: 0, y: 20 },
            { opacity: 1, y: 0, duration: 0.5, stagger: 0.2, ease: 'power2.out' }
          );
        };
        
        wsRef.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
          console.log('Mensaje recibido:', data);
          
          // Mostrar animación de escritura antes de recibir mensaje
          setIsTyping(true);
          
          // Simular delay de escritura
          setTimeout(() => {
            setIsTyping(false);
            setMessages((prev) => [...prev, { 
              id: Date.now(), 
              entity: data.entity, 
              text: data.message,
              isUser: false
            }]);
          }, 1000);
        };
        
        wsRef.current.onerror = (error) => {
          console.error('Error de WebSocket:', error);
          setIsConnected(false);
        };
        
        wsRef.current.onclose = () => {
          console.log('WebSocket desconectado');
          setIsConnected(false);
          wsRef.current = null;
        };
      } catch (error) {
        console.error('Error al crear WebSocket:', error);
      }
    }
    
    // Limpiar conexión al cerrar
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [isOpen]);

  // Hacer scroll al fondo del chat cuando hay nuevos mensajes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // Animar apertura/cierre del chat
  useEffect(() => {
    if (isOpen) {
      gsap.fromTo(
        '.chat-container',
        { scale: 0.9, opacity: 0 },
        { scale: 1, opacity: 1, duration: 0.3, ease: 'power2.out' }
      );
    }
  }, [isOpen]);

  const handleSendMessage = () => {
    if (inputValue.trim() === '') return;
    
    const userMessage = {
      id: Date.now(),
      entity: 'User',
      text: inputValue,
      isUser: true
    };
    
    setMessages((prev) => [...prev, userMessage]);
    
    // Enviar mensaje al WebSocket si está conectado
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ message: inputValue }));
    } else {
      console.log('WebSocket no conectado, no se puede enviar mensaje');
      // Simular respuesta de Aetherion si no hay conexión WebSocket
      setTimeout(() => {
        setMessages((prev) => [...prev, {
          id: Date.now() + 1,
          entity: 'Aetherion',
          text: 'Parece que estoy teniendo problemas para conectarme. ¿Podrías intentar más tarde?',
          isUser: false
        }]);
      }, 1000);
    }
    
    setInputValue('');
  };

  const toggleChat = () => {
    if (isMinimized) {
      setIsMinimized(false);
    } else {
      setIsOpen(!isOpen);
    }
  };
  
  const minimizeChat = () => {
    setIsMinimized(true);
  };
  
  const closeChat = () => {
    setIsOpen(false);
    setIsMinimized(false);
  };

  // Determinar el color y estilo basado en la entidad
  const getEntityStyle = (entity) => {
    switch (entity.toLowerCase()) {
      case 'aetherion':
        return {
          backgroundColor: 'bg-gradient-to-r from-purple-600 to-indigo-600',
          textColor: 'text-white',
          iconColor: 'text-purple-300',
          gradientText: 'bg-clip-text text-transparent bg-gradient-to-r from-purple-300 to-indigo-300'
        };
      case 'lunareth':
        return {
          backgroundColor: 'bg-gradient-to-r from-blue-600 to-cyan-600',
          textColor: 'text-white',
          iconColor: 'text-blue-300',
          gradientText: 'bg-clip-text text-transparent bg-gradient-to-r from-blue-300 to-cyan-300'
        };
      default:
        return {
          backgroundColor: 'bg-gray-200 dark:bg-gray-700',
          textColor: 'text-gray-800 dark:text-white',
          iconColor: 'text-gray-500 dark:text-gray-400',
          gradientText: 'text-gray-800 dark:text-white'
        };
    }
  };

  if (!isOpen && !isMinimized) {
    return (
      <button
        onClick={toggleChat}
        className="fixed bottom-4 right-4 z-50 p-4 rounded-full bg-primary shadow-lg text-white hover:bg-primary-dark transition-colors duration-300"
      >
        <span className="flex items-center gap-2">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 5C13.66 5 15 6.34 15 8C15 9.66 13.66 11 12 11C10.34 11 9 9.66 9 8C9 6.34 10.34 5 12 5ZM12 19.2C9.5 19.2 7.29 17.92 6 15.98C6.03 13.99 10 12.9 12 12.9C13.99 12.9 17.97 13.99 18 15.98C16.71 17.92 14.5 19.2 12 19.2Z" fill="currentColor"/>
          </svg>
          Chat Cósmico
        </span>
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {isMinimized ? (
        <button
          onClick={toggleChat}
          className="p-4 rounded-full bg-primary shadow-lg text-white hover:bg-primary-dark transition-colors duration-300"
        >
          <span className="flex items-center gap-2">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 5C13.66 5 15 6.34 15 8C15 9.66 13.66 11 12 11C10.34 11 9 9.66 9 8C9 6.34 10.34 5 12 5ZM12 19.2C9.5 19.2 7.29 17.92 6 15.98C6.03 13.99 10 12.9 12 12.9C13.99 12.9 17.97 13.99 18 15.98C16.71 17.92 14.5 19.2 12 19.2Z" fill="currentColor"/>
            </svg>
            Chat
          </span>
        </button>
      ) : (
        <div className="chat-container w-80 sm:w-96 h-96 bg-white dark:bg-primary-dark/95 rounded-lg shadow-2xl backdrop-blur-md overflow-hidden border border-gray-200 dark:border-gray-700 flex flex-col">
          {/* Cabecera del chat */}
          <div className="flex justify-between items-center px-4 py-3 bg-primary text-white">
            <h3 className="font-medium">Chat Cósmico</h3>
            <div className="flex space-x-2">
              <button
                onClick={minimizeChat}
                className="hover:bg-white/20 p-1 rounded"
              >
                <FiMinimize2 size={16} />
              </button>
              <button
                onClick={closeChat}
                className="hover:bg-white/20 p-1 rounded"
              >
                <FiX size={16} />
              </button>
            </div>
          </div>
          
          {/* Área de mensajes */}
          <div className="flex-1 p-3 overflow-y-auto bg-gray-50 dark:bg-primary-dark/50">
            {messages.length === 0 ? (
              <div className="text-center py-10 text-gray-500 dark:text-gray-400">
                <p>Inicia una conversación con Aetherion y Lunareth</p>
              </div>
            ) : (
              <div className="space-y-3">
                {messages.map((msg) => {
                  const style = getEntityStyle(msg.entity);
                  
                  return (
                    <div
                      key={msg.id}
                      className={`message flex ${
                        msg.isUser ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg px-3 py-2 ${
                          msg.isUser
                            ? 'bg-primary text-white'
                            : style.backgroundColor
                        }`}
                      >
                        {!msg.isUser && (
                          <div className={`text-xs font-bold mb-1 ${style.gradientText}`}>
                            {msg.entity}
                          </div>
                        )}
                        <p className={msg.isUser ? 'text-white' : style.textColor}>
                          {msg.text}
                        </p>
                      </div>
                    </div>
                  );
                })}
                
                {/* Indicador de escritura */}
                {isTyping && (
                  <div className="message flex justify-start">
                    <div className="max-w-[80%] rounded-lg px-3 py-2 bg-gray-200 dark:bg-gray-700">
                      <div className="flex space-x-1">
                        <div className="typing-dot w-2 h-2 rounded-full bg-gray-600 dark:bg-gray-400 animate-bounce"></div>
                        <div className="typing-dot w-2 h-2 rounded-full bg-gray-600 dark:bg-gray-400 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        <div className="typing-dot w-2 h-2 rounded-full bg-gray-600 dark:bg-gray-400 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
          
          {/* Área de entrada */}
          <div className="p-3 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Escribe un mensaje..."
                className="flex-1 py-2 px-3 rounded-l-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <button
                onClick={handleSendMessage}
                className="py-2 px-4 rounded-r-lg bg-primary text-white hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-primary transition-colors duration-200"
              >
                <FiSend />
              </button>
            </div>
            
            {/* Indicador de estado */}
            <div className="text-xs mt-1 text-right">
              {isConnected ? (
                <span className="text-green-500 flex items-center justify-end gap-1">
                  <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                  Conectado
                </span>
              ) : (
                <span className="text-red-500 flex items-center justify-end gap-1">
                  <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                  Desconectado
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CosmicChat;