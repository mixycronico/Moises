import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  FiTrendingUp, FiTrendingDown, FiDollarSign, 
  FiBarChart, FiActivity, FiRefreshCw, FiClock
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

const Dashboard = ({ user }) => {
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState(null);
  const [error, setError] = useState(null);
  
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
  
  return (
    <motion.div
      className="p-4 md:p-6"
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      {/* Título del dashboard */}
      <motion.div 
        className="mb-8"
        variants={itemVariants}
      >
        <h1 className="text-2xl md:text-3xl font-display mb-2">
          Dashboard <span className="text-cosmic-glow">del Inversionista</span>
        </h1>
        <p className="text-gray-400">
          Bienvenido, <span className="text-white font-medium">{user?.username || 'Usuario'}</span>. 
          Tu categoría actual es <span className="font-medium text-cosmic-yellow">
            {dashboardData.category.charAt(0).toUpperCase() + dashboardData.category.slice(1)}
          </span>
        </p>
      </motion.div>
      
      {/* Tarjetas principales */}
      <motion.div 
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6"
        variants={itemVariants}
      >
        {/* Balance */}
        <div className="cosmic-card p-5">
          <h3 className="text-sm text-gray-400 mb-1 flex items-center">
            <FiDollarSign className="mr-1" /> Balance Actual
          </h3>
          <p className="text-2xl font-semibold cosmic-glow-text">
            {formatCurrency(dashboardData.balance)}
          </p>
          <div className="mt-2 text-sm flex items-center">
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
        
        {/* Capital */}
        <div className="cosmic-card p-5">
          <h3 className="text-sm text-gray-400 mb-1 flex items-center">
            <FiBarChart className="mr-1" /> Capital Invertido
          </h3>
          <p className="text-2xl font-semibold">
            {formatCurrency(dashboardData.capital)}
          </p>
          <p className="mt-2 text-sm text-gray-400">
            Fecha de inicio: {formatDate('2024-06-15')}
          </p>
        </div>
        
        {/* Ganancias */}
        <div className="cosmic-card p-5">
          <h3 className="text-sm text-gray-400 mb-1 flex items-center">
            <FiActivity className="mr-1" /> Ganancias Totales
          </h3>
          <p className="text-2xl font-semibold text-cosmic-green">
            +{formatCurrency(dashboardData.earnings)}
          </p>
          <div className="mt-2 text-sm">
            <span className="text-cosmic-green flex items-center">
              <FiTrendingUp className="mr-1" />
              +{dashboardData.earningsPercentage}% total
            </span>
          </div>
        </div>
        
        {/* Próxima predicción */}
        <div className="cosmic-card p-5">
          <h3 className="text-sm text-gray-400 mb-1 flex items-center">
            <FiClock className="mr-1" /> Próxima Predicción
          </h3>
          <p className="text-xl font-semibold">
            {formatDate(dashboardData.nextPrediction)}
          </p>
          <p className="mt-2 text-sm text-gray-400">
            {formatTime(dashboardData.nextPrediction)}
          </p>
        </div>
      </motion.div>
      
      {/* Gráfico y transacciones */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Gráfico de rendimiento */}
        <motion.div 
          className="lg:col-span-2 cosmic-card p-5"
          variants={itemVariants}
        >
          <h3 className="text-lg font-semibold mb-4">Rendimiento Anual</h3>
          <div className="h-64">
            {dashboardData.performanceData && (
              <Line
                data={dashboardData.performanceData}
                options={chartOptions}
              />
            )}
          </div>
        </motion.div>
        
        {/* Transacciones recientes */}
        <motion.div 
          className="cosmic-card p-5"
          variants={itemVariants}
        >
          <h3 className="text-lg font-semibold mb-4">Transacciones Recientes</h3>
          <div className="space-y-3">
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
        </motion.div>
      </div>
      
      {/* Activos */}
      <motion.div 
        className="cosmic-card p-5 mb-6"
        variants={itemVariants}
      >
        <h3 className="text-lg font-semibold mb-4">Activos en Cartera</h3>
        <div className="overflow-x-auto">
          <table className="w-full min-w-full table-auto">
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
      </motion.div>
      
      {/* Estado del sistema */}
      <motion.div 
        className="cosmic-card p-5"
        variants={itemVariants}
      >
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Estado del Sistema</h3>
          <div className={`flex items-center ${dashboardData.systemStatus.status === 'online' ? 'text-cosmic-green' : 'text-cosmic-yellow'}`}>
            <span className="inline-block w-2 h-2 rounded-full bg-current mr-2"></span>
            <span>
              {dashboardData.systemStatus.status === 'online' ? 'En línea' : 'Mantenimiento'}
            </span>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-3 bg-cosmic-primary/10 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Precisión de Predicciones</p>
            <p className="text-xl font-semibold text-cosmic-glow">{dashboardData.systemStatus.predictionAccuracy}%</p>
          </div>
          
          <div className="text-center p-3 bg-cosmic-primary/10 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Última Actualización</p>
            <p className="text-xl font-semibold">{formatTime(dashboardData.systemStatus.lastUpdated)}</p>
          </div>
          
          <div className="text-center p-3 bg-cosmic-primary/10 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Modo Sistema</p>
            <p className="text-xl font-semibold text-cosmic-highlight">Quantum Ultra</p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default Dashboard;