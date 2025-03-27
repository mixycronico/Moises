import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  FiTrendingUp, FiTrendingDown, FiDollarSign, 
  FiBarChart, FiActivity, FiRefreshCw, FiClock,
  FiMove, FiLock, FiUnlock
} from 'react-icons/fi';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Responsive, WidthProvider } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

// Registrar componentes de ChartJS
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Crear el componente responsive
const ResponsiveGridLayout = WidthProvider(Responsive);

const Dashboard = ({ user }) => {
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState(null);
  const [error, setError] = useState(null);
  const [isEditMode, setIsEditMode] = useState(false);
  
  // Definir layouts para diferentes tamaños de pantalla
  const [layouts, setLayouts] = useState({
    lg: [
      { i: 'balance', x: 0, y: 0, w: 3, h: 1 },
      { i: 'transactions', x: 0, y: 1, w: 3, h: 2 },
      { i: 'performance', x: 3, y: 0, w: 9, h: 3 },
      { i: 'assets', x: 0, y: 3, w: 12, h: 2 },
      { i: 'system', x: 0, y: 5, w: 12, h: 1 }
    ],
    md: [
      { i: 'balance', x: 0, y: 0, w: 4, h: 1 },
      { i: 'transactions', x: 0, y: 1, w: 4, h: 2 },
      { i: 'performance', x: 4, y: 0, w: 8, h: 3 },
      { i: 'assets', x: 0, y: 3, w: 12, h: 2 },
      { i: 'system', x: 0, y: 5, w: 12, h: 1 }
    ],
    sm: [
      { i: 'balance', x: 0, y: 0, w: 6, h: 1 },
      { i: 'transactions', x: 6, y: 0, w: 6, h: 2 },
      { i: 'performance', x: 0, y: 1, w: 12, h: 2 },
      { i: 'assets', x: 0, y: 3, w: 12, h: 2 },
      { i: 'system', x: 0, y: 5, w: 12, h: 1 }
    ],
    xs: [
      { i: 'balance', x: 0, y: 0, w: 12, h: 1 },
      { i: 'transactions', x: 0, y: 1, w: 12, h: 2 },
      { i: 'performance', x: 0, y: 3, w: 12, h: 2 },
      { i: 'assets', x: 0, y: 5, w: 12, h: 2 },
      { i: 'system', x: 0, y: 7, w: 12, h: 1 }
    ]
  });
  
  useEffect(() => {
    // Cargar datos del dashboard
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        const response = await axios.get('/api/investor/dashboard');
        
        if (response.data.success) {
          setDashboardData(response.data.data);
        } else {
          setError(response.data.message || 'Error al cargar los datos del dashboard');
        }
      } catch (error) {
        console.error('Error loading dashboard data:', error);
        setError('No se pudo conectar con el servidor. Por favor, intenta de nuevo más tarde.');
        
        // Datos simulados para desarrollo (se eliminarían en producción)
        setDashboardData({
          balance: 12586.42,
          capital: 10000,
          earnings: 2586.42,
          earningsPercentage: 25.86,
          todayChange: 125.32,
          todayPercentage: 1.01,
          status: 'active',
          category: 'silver',
          recentTransactions: [
            { id: 1, type: 'profit', amount: 125.32, date: '2025-03-27T10:23:45Z', description: 'BTC/USDT' },
            { id: 2, type: 'profit', amount: 85.67, date: '2025-03-26T14:15:22Z', description: 'ETH/USDT' },
            { id: 3, type: 'deposit', amount: 1000, date: '2025-03-25T09:30:00Z', description: 'Depósito mensual' },
            { id: 4, type: 'profit', amount: 42.18, date: '2025-03-24T16:45:30Z', description: 'SOL/USDT' },
          ],
          performanceData: {
            labels: ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
            datasets: [
              {
                label: 'Rendimiento 2025',
                data: [0, 3.2, 7.5, 12.8, 15.6, 18.2, 22.5, 25.86],
                borderColor: 'rgba(158, 107, 219, 1)',
                backgroundColor: 'rgba(158, 107, 219, 0.1)',
                fill: true,
                tension: 0.4,
              }
            ]
          },
          assets: [
            { symbol: 'BTC', name: 'Bitcoin', amount: 0.15, value: 6250.25, change: 2.34 },
            { symbol: 'ETH', name: 'Ethereum', amount: 2.5, value: 4320.75, change: 1.56 },
            { symbol: 'SOL', name: 'Solana', amount: 12, value: 1985.42, change: -0.78 },
          ],
          nextPrediction: '2025-03-28T09:00:00Z',
          systemStatus: {
            status: 'online',
            predictionAccuracy: 94.2,
            lastUpdated: '2025-03-27T08:15:00Z',
          }
        });
      } finally {
        setLoading(false);
      }
    };
    
    fetchDashboardData();
  }, []);
  
  // Opciones para gráficos
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(13, 13, 33, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(158, 107, 219, 0.3)',
        borderWidth: 1,
        padding: 10,
        displayColors: false,
        callbacks: {
          label: function(context) {
            return `Rendimiento: ${context.parsed.y}%`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false,
          drawBorder: false,
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
        }
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
          drawBorder: false,
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
          callback: function(value) {
            return value + '%';
          }
        }
      }
    },
    elements: {
      point: {
        radius: 2,
        hoverRadius: 5,
        backgroundColor: 'rgba(158, 107, 219, 1)',
        borderColor: 'rgba(255, 255, 255, 0.8)',
      }
    }
  };
  
  // Variantes de animación
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.4 }
    }
  };
  
  // Formatear moneda
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('es-ES', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };
  
  // Formatear fecha
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('es-ES', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric'
    });
  };
  
  // Formatear hora
  const formatTime = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString('es-ES', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };
  
  // Determinar el estado de tendencia
  const getTrendStatus = (value) => {
    return value >= 0 ? {
      color: 'text-cosmic-green',
      icon: <FiTrendingUp className="mr-1" />
    } : {
      color: 'text-cosmic-red',
      icon: <FiTrendingDown className="mr-1" />
    };
  };
  
  // Función para guardar el layout
  const handleLayoutChange = (currentLayout, allLayouts) => {
    if (isEditMode) {
      setLayouts(allLayouts);
      
      // Opcional: guardar en localStorage para persistencia
      localStorage.setItem('dashboardLayouts', JSON.stringify(allLayouts));
    }
  };
  
  // Función para alternar el modo de edición
  const toggleEditMode = () => {
    const newEditMode = !isEditMode;
    setIsEditMode(newEditMode);
    
    // En móvil, mostrar instrucciones cuando se activa el modo edición
    if (newEditMode && window.innerWidth < 768) {
      // Pequeña vibración para feedback táctil si está disponible
      if (navigator.vibrate) {
        navigator.vibrate(200);
      }
      
      // Mostrar toast o alerta con instrucciones - Esto es muy básico, idealmente usarías un componente Toast
      const toast = document.createElement('div');
      toast.className = 'fixed bottom-20 left-1/2 transform -translate-x-1/2 bg-cosmic-primary-50 text-white px-4 py-3 rounded-lg shadow-lg z-50 text-sm flex items-center';
      toast.style.maxWidth = '90%';
      toast.style.width = '320px';
      toast.innerHTML = `
        <div class="mr-3 text-cosmic-accent">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="16" x2="12" y2="12"></line>
            <line x1="12" y1="8" x2="12.01" y2="8"></line>
          </svg>
        </div>
        <div>
          Arrastra los paneles desde los iconos <span class="text-cosmic-accent">⋮⋮</span> para reordenarlos
        </div>
      `;
      
      document.body.appendChild(toast);
      
      // Eliminar después de 4 segundos
      setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.5s';
        setTimeout(() => {
          document.body.removeChild(toast);
        }, 500);
      }, 4000);
    }
  };
  
  // Cargar layouts guardados al iniciar
  useEffect(() => {
    const savedLayouts = localStorage.getItem('dashboardLayouts');
    if (savedLayouts) {
      try {
        setLayouts(JSON.parse(savedLayouts));
      } catch (error) {
        console.error('Error al cargar layouts guardados:', error);
      }
    }
  }, []);
  
  if (loading) {
    return (
      <div className="flex justify-center items-center h-[calc(100vh-4rem)]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cosmic-accent"></div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <div className="bg-cosmic-red/10 border border-cosmic-red/30 text-cosmic-red rounded-md p-4 text-center">
          <p>{error}</p>
          <button
            className="mt-4 cosmic-button-secondary"
            onClick={() => window.location.reload()}
          >
            <FiRefreshCw className="mr-2" />
            Reintentar
          </button>
        </div>
      </div>
    );
  }
  
  // Componente para la tarjeta del panel
  const DashboardCard = ({ title, children, className = "", dragHandle = false }) => {
    return (
      <div className={`cosmic-card relative ${className} h-full overflow-hidden`}>
        {dragHandle && isEditMode && (
          <div className="absolute top-2 right-2 z-10 cursor-move p-1 rounded-full bg-cosmic-primary/50 text-cosmic-glow">
            <FiMove size={14} />
          </div>
        )}
        <div className="p-4 h-full flex flex-col">
          <h3 className="text-lg font-semibold mb-4 flex items-center">{title}</h3>
          <div className="flex-1">{children}</div>
        </div>
      </div>
    );
  };

  // Componentes específicos para cada panel
  const BalanceCard = () => (
    <div className="flex flex-col h-full p-1">
      <p className="text-2xl font-semibold cosmic-glow-text">
        {formatCurrency(dashboardData.balance)}
      </p>
      <div className="mt-auto text-sm flex items-center">
        {dashboardData.todayPercentage >= 0 ? (
          <span className="flex items-center text-cosmic-green">
            <FiTrendingUp className="mr-1" />
            +{dashboardData.todayPercentage}% hoy
          </span>
        ) : (
          <span className="flex items-center text-cosmic-red">
            <FiTrendingDown className="mr-1" />
            {dashboardData.todayPercentage}% hoy
          </span>
        )}
        <span className="ml-2 text-gray-500">
          ({formatCurrency(dashboardData.todayChange)})
        </span>
      </div>
    </div>
  );

  const PerformanceCard = () => (
    <div className="h-full">
      {dashboardData.performanceData && (
        <Line data={dashboardData.performanceData} options={chartOptions} />
      )}
    </div>
  );

  const TransactionsCard = () => (
    <div className="space-y-3 flex flex-col h-full">
      <div className="flex-1 overflow-auto">
        {dashboardData.recentTransactions.map((transaction) => {
          // Determinar icono y color según tipo de transacción
          let icon, colorClass;
          switch (transaction.type) {
            case 'profit':
              icon = <FiTrendingUp />;
              colorClass = 'text-cosmic-green';
              break;
            case 'loss':
              icon = <FiTrendingDown />;
              colorClass = 'text-cosmic-red';
              break;
            case 'deposit':
              icon = <FiDollarSign />;
              colorClass = 'text-cosmic-blue';
              break;
            case 'withdrawal':
              icon = <FiDollarSign />;
              colorClass = 'text-cosmic-yellow';
              break;
            default:
              icon = <FiActivity />;
              colorClass = 'text-cosmic-glow';
          }
          
          return (
            <div key={transaction.id} className="flex items-center justify-between py-2 border-b border-cosmic-primary/10 last:border-0">
              <div className="flex items-center">
                <div className={`w-8 h-8 rounded-full bg-cosmic-primary/20 flex items-center justify-center ${colorClass} mr-3`}>
                  {icon}
                </div>
                <div>
                  <p className="font-medium">{transaction.description}</p>
                  <p className="text-xs text-gray-400">{formatDate(transaction.date)}</p>
                </div>
              </div>
              <div className={transaction.type === 'profit' ? 'text-cosmic-green' : (transaction.type === 'loss' ? 'text-cosmic-red' : '')}>
                {transaction.type === 'profit' && '+'}
                {formatCurrency(transaction.amount)}
              </div>
            </div>
          );
        })}
      </div>
      <button className="w-full mt-4 cosmic-button-secondary text-sm">
        Ver todas las transacciones
      </button>
    </div>
  );

  const AssetsCard = () => (
    <div className="overflow-x-auto -mx-4 px-4 pb-2">
      <table className="w-full min-w-[500px] table-auto">
        <thead>
          <tr className="text-left text-gray-400 border-b border-cosmic-primary/20">
            <th className="px-4 py-2">Activo</th>
            <th className="px-4 py-2">Cantidad</th>
            <th className="px-4 py-2">Valor</th>
            <th className="px-4 py-2">Cambio 24h</th>
          </tr>
        </thead>
        <tbody>
          {dashboardData.assets.map((asset) => {
            const trendStatus = getTrendStatus(asset.change);
            
            return (
              <tr key={asset.symbol} className="border-b border-cosmic-primary/10 last:border-0">
                <td className="px-4 py-3">
                  <div className="flex items-center">
                    <div className="w-8 h-8 rounded-full bg-cosmic-primary/20 flex items-center justify-center text-cosmic-glow mr-3">
                      {asset.symbol.charAt(0)}
                    </div>
                    <div>
                      <p className="font-medium">{asset.name}</p>
                      <p className="text-xs text-gray-400">{asset.symbol}</p>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3">{asset.amount}</td>
                <td className="px-4 py-3">{formatCurrency(asset.value)}</td>
                <td className="px-4 py-3">
                  <span className={`flex items-center ${trendStatus.color}`}>
                    {trendStatus.icon}
                    {asset.change > 0 ? '+' : ''}{asset.change}%
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );

  const SystemCard = () => (
    <div>
      <div className="flex justify-between items-center mb-4">
        <div className={`flex items-center ${dashboardData.systemStatus.status === 'online' ? 'text-cosmic-green' : 'text-cosmic-yellow'}`}>
          <span className="inline-block w-2 h-2 rounded-full bg-current mr-2"></span>
          <span>
            {dashboardData.systemStatus.status === 'online' ? 'En línea' : 'Mantenimiento'}
          </span>
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center p-3 bg-cosmic-primary/10 rounded-lg">
          <p className="text-sm text-gray-400 mb-1">Precisión</p>
          <p className="text-xl font-semibold text-cosmic-glow">{dashboardData.systemStatus.predictionAccuracy}%</p>
        </div>
        
        <div className="text-center p-3 bg-cosmic-primary/10 rounded-lg">
          <p className="text-sm text-gray-400 mb-1">Actualización</p>
          <p className="text-xl font-semibold">{formatTime(dashboardData.systemStatus.lastUpdated)}</p>
        </div>
        
        <div className="text-center p-3 bg-cosmic-primary/10 rounded-lg">
          <p className="text-sm text-gray-400 mb-1">Modo</p>
          <p className="text-xl font-semibold text-cosmic-highlight">Quantum</p>
        </div>
      </div>
    </div>
  );

  return (
    <motion.div
      className="p-4 md:p-6 pb-28 md:pb-6"
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      {/* Título del dashboard */}
      <motion.div 
        className="mb-8 flex flex-col md:flex-row justify-between md:items-center"
        variants={itemVariants}
      >
        <div>
          <h1 className="text-2xl md:text-3xl font-display mb-2">
            Dashboard <span className="text-cosmic-glow">del Inversionista</span>
          </h1>
          <p className="text-gray-400 mb-2 md:mb-0">
            Bienvenido, <span className="text-white font-medium">{user?.username || 'Usuario'}</span>. 
            Tu categoría actual es <span className="font-medium text-cosmic-yellow">
              {dashboardData.category.charAt(0).toUpperCase() + dashboardData.category.slice(1)}
            </span>
          </p>
        </div>
        
        {/* Botón para editar layout */}
        <button 
          onClick={toggleEditMode}
          className={`flex items-center space-x-2 ${isEditMode ? 'cosmic-button' : 'cosmic-button-secondary'}`}
        >
          {isEditMode ? (
            <>
              <FiLock className="mr-1" />
              <span>Guardar Posiciones</span>
            </>
          ) : (
            <>
              <FiUnlock className="mr-1" />
              <span>Personalizar Paneles</span>
            </>
          )}
        </button>
      </motion.div>
      
      {/* Versión móvil: carruseles desplazables y reorganizables */}
      <div className={`md:hidden mb-6 ${isEditMode ? 'edit-mode' : ''}`}>
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-lg font-semibold">Resumen</h2>
          <button 
            onClick={toggleEditMode}
            className={`text-xs flex items-center py-1.5 px-3 rounded-full ${isEditMode ? 'bg-cosmic-accent text-white' : 'bg-cosmic-primary-50 text-cosmic-glow border border-cosmic-accent/30'}`}
          >
            {isEditMode ? (
              <>
                <FiLock className="mr-1" /> Guardar
              </>
            ) : (
              <>
                <FiUnlock className="mr-1" /> Personalizar
              </>
            )}
          </button>
        </div>
        
        <ResponsiveGridLayout
          className="layout"
          layouts={{ xs: layouts.xs }}
          breakpoints={{ xs: 480 }}
          cols={{ xs: 1 }}
          rowHeight={150}
          isDraggable={isEditMode}
          isResizable={false}
          draggableHandle=".mobile-drag-handle"
          onLayoutChange={(layout, allLayouts) => {
            if (isEditMode && layout) {
              const newLayouts = { ...layouts, xs: layout };
              setLayouts(newLayouts);
              localStorage.setItem('dashboardLayouts', JSON.stringify(newLayouts));
            }
          }}
          margin={[10, 15]}
          containerPadding={[0, 5]}
          useCSSTransforms={true}
        >
          <div key="balance" className="min-h-[120px]">
            <div className="cosmic-card p-5 h-full relative">
              {isEditMode && (
                <div className="mobile-drag-handle flex items-center justify-center">
                  <FiMove size={16} className="text-white" />
                </div>
              )}
              <h3 className="text-sm text-gray-400 mb-1 flex items-center">
                <FiDollarSign className="mr-1" /> Balance Actual
              </h3>
              <BalanceCard />
            </div>
          </div>
          
          <div key="performance" className="min-h-[200px]">
            <div className="cosmic-card p-5 h-full relative">
              {isEditMode && (
                <div className="mobile-drag-handle flex items-center justify-center">
                  <FiMove size={16} className="text-white" />
                </div>
              )}
              <h3 className="text-sm text-gray-400 mb-1 flex items-center">
                <FiActivity className="mr-1" /> Rendimiento
              </h3>
              <div className="h-36">
                <PerformanceCard />
              </div>
            </div>
          </div>
          
          <div key="system" className="min-h-[200px]">
            <div className="cosmic-card p-5 h-full relative">
              {isEditMode && (
                <div className="mobile-drag-handle flex items-center justify-center">
                  <FiMove size={16} className="text-white" />
                </div>
              )}
              <h3 className="text-sm text-gray-400 mb-1 flex items-center">
                <FiBarChart className="mr-1" /> Estado del Sistema
              </h3>
              <SystemCard />
            </div>
          </div>
          
          <div key="transactions" className="min-h-[300px]">
            <div className="cosmic-card p-5 h-full relative">
              {isEditMode && (
                <div className="mobile-drag-handle flex items-center justify-center">
                  <FiMove size={16} className="text-white" />
                </div>
              )}
              <h3 className="text-sm text-gray-400 mb-1 flex items-center">
                <FiActivity className="mr-1" /> Transacciones Recientes
              </h3>
              <div className="h-64 overflow-auto">
                <TransactionsCard />
              </div>
            </div>
          </div>
          
          <div key="assets" className="min-h-[250px]">
            <div className="cosmic-card p-5 h-full relative">
              {isEditMode && (
                <div className="mobile-drag-handle flex items-center justify-center">
                  <FiMove size={16} className="text-white" />
                </div>
              )}
              <h3 className="text-sm text-gray-400 mb-1 flex items-center">
                <FiDollarSign className="mr-1" /> Activos en Cartera
              </h3>
              <div className="h-48 overflow-auto">
                <AssetsCard />
              </div>
            </div>
          </div>
        </ResponsiveGridLayout>
      </div>
      
      {/* Versión desktop: grid layout modular */}
      <div className={`hidden md:block ${isEditMode ? 'edit-mode' : ''}`}>
        <ResponsiveGridLayout
          className="layout"
          layouts={layouts}
          breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480 }}
          cols={{ lg: 12, md: 12, sm: 12, xs: 12 }}
          rowHeight={80}
          onLayoutChange={handleLayoutChange}
          isDraggable={isEditMode}
          isResizable={isEditMode}
          draggableHandle=".cursor-move"
        >
          <div key="balance">
            <DashboardCard title={<><FiDollarSign className="mr-2" /> Balance Actual</>} dragHandle={true}>
              <BalanceCard />
            </DashboardCard>
          </div>
          
          <div key="performance">
            <DashboardCard title={<><FiBarChart className="mr-2" /> Rendimiento Anual</>} dragHandle={true} className="overflow-hidden">
              <PerformanceCard />
            </DashboardCard>
          </div>
          
          <div key="transactions">
            <DashboardCard title={<><FiActivity className="mr-2" /> Transacciones Recientes</>} dragHandle={true}>
              <TransactionsCard />
            </DashboardCard>
          </div>
          
          <div key="assets">
            <DashboardCard title={<><FiDollarSign className="mr-2" /> Activos en Cartera</>} dragHandle={true}>
              <AssetsCard />
            </DashboardCard>
          </div>
          
          <div key="system">
            <DashboardCard title={<><FiClock className="mr-2" /> Estado del Sistema</>} dragHandle={true}>
              <SystemCard />
            </DashboardCard>
          </div>
        </ResponsiveGridLayout>
      </div>
    </motion.div>
  );
};

export default Dashboard;