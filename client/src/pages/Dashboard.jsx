import { useState, useEffect } from 'react';
import { FiTrendingUp, FiDollarSign, FiBarChart, FiArrowUp, FiArrowDown, FiActivity, FiClock } from 'react-icons/fi';
import { Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import MainLayout from '../components/MainLayout';
import gsap from 'gsap';

// Registrar componentes de Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState(null);

  // Obtener datos del dashboard
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/api/investor/dashboard');
        
        if (!response.ok) {
          throw new Error('Error al cargar los datos del dashboard');
        }
        
        const data = await response.json();
        setDashboardData(data);
      } catch (error) {
        console.error('Error:', error);
        setErrorMessage(error.message);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, []);

  // Animar elementos al cargar
  useEffect(() => {
    if (!isLoading && dashboardData) {
      // Animación de las tarjetas de estadísticas
      gsap.fromTo(
        '.stat-card',
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.5, stagger: 0.1, ease: 'power2.out' }
      );
      
      // Animación de los gráficos
      gsap.fromTo(
        '.chart-container',
        { opacity: 0, scale: 0.95 },
        { opacity: 1, scale: 1, duration: 0.7, delay: 0.3, stagger: 0.2, ease: 'back.out(1.7)' }
      );
      
      // Animación de las transacciones
      gsap.fromTo(
        '.transaction-item',
        { opacity: 0, x: -20 },
        { opacity: 1, x: 0, duration: 0.5, delay: 0.5, stagger: 0.1, ease: 'power2.out' }
      );
    }
  }, [isLoading, dashboardData]);

  // Datos para el gráfico de línea
  const lineChartData = {
    labels: ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
    datasets: [
      {
        label: 'Capital',
        data: [20000, 20500, 21300, 22100, 22800, 23500, 24100, 24800, 25000, 26000, 26500, 27200],
        borderColor: '#6366F1',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        tension: 0.4,
        fill: true,
      },
      {
        label: 'Ganancias',
        data: [0, 500, 1300, 2100, 2800, 3500, 4100, 4800, 5000, 6000, 6500, 7200],
        borderColor: '#4FFBDF',
        backgroundColor: 'rgba(79, 251, 223, 0.1)',
        tension: 0.4,
        fill: true,
      },
    ],
  };

  // Opciones para el gráfico de línea
  const lineChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: 'rgba(255, 255, 255, 0.8)',
          font: {
            family: 'Inter, sans-serif',
          },
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(17, 24, 39, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(99, 102, 241, 0.2)',
        borderWidth: 1,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
          callback: function(value) {
            return '$' + value.toLocaleString();
          },
        },
      },
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
        },
      },
    },
  };

  // Datos para el gráfico de dona
  const doughnutChartData = {
    labels: ['Capital', 'Rendimientos', 'Bonos'],
    datasets: [
      {
        data: [70, 25, 5],
        backgroundColor: [
          'rgba(99, 102, 241, 0.8)',
          'rgba(79, 251, 223, 0.8)',
          'rgba(248, 113, 113, 0.8)',
        ],
        borderColor: [
          'rgba(99, 102, 241, 1)',
          'rgba(79, 251, 223, 1)',
          'rgba(248, 113, 113, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Opciones para el gráfico de dona
  const doughnutChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: 'rgba(255, 255, 255, 0.8)',
          font: {
            family: 'Inter, sans-serif',
          },
        },
      },
      tooltip: {
        backgroundColor: 'rgba(17, 24, 39, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(99, 102, 241, 0.2)',
        borderWidth: 1,
      },
    },
    cutout: '70%',
  };

  // Renderizar estado de carga o error
  if (isLoading) {
    return (
      <MainLayout title="Dashboard">
        <div className="flex items-center justify-center h-full">
          <div className="flex flex-col items-center">
            <div className="w-16 h-16 border-t-4 border-b-4 border-secondary rounded-full animate-spin"></div>
            <p className="mt-4 text-gray-500 dark:text-gray-400">Cargando datos...</p>
          </div>
        </div>
      </MainLayout>
    );
  }

  if (errorMessage) {
    return (
      <MainLayout title="Dashboard">
        <div className="flex items-center justify-center h-full">
          <div className="p-6 max-w-md bg-red-500/10 rounded-lg border border-red-500/30 text-center">
            <div className="text-red-500 text-5xl mb-4">⚠️</div>
            <h3 className="text-xl font-semibold text-red-500 mb-2">Error al cargar datos</h3>
            <p className="text-gray-500 dark:text-gray-400">{errorMessage}</p>
            <button
              className="mt-4 px-4 py-2 bg-secondary rounded-md text-white hover:bg-secondary-light transition-colors"
              onClick={() => window.location.reload()}
            >
              Intentar nuevamente
            </button>
          </div>
        </div>
      </MainLayout>
    );
  }

  return (
    <MainLayout title="Dashboard">
      {/* Tarjetas de estadísticas principales */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="stat-card p-6 bg-white/10 dark:bg-primary-light/5 backdrop-blur-md rounded-xl border border-white/10 dark:border-primary-light/10 shadow-sm">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Capital Total</p>
              <h3 className="text-2xl font-bold text-primary-dark dark:text-white mt-1">
                ${dashboardData?.capital.toLocaleString()}
              </h3>
            </div>
            <div className="p-2 bg-indigo-100 dark:bg-indigo-500/20 rounded-lg text-indigo-600 dark:text-indigo-400">
              <FiDollarSign size={24} />
            </div>
          </div>
          <div className="mt-4 flex items-center">
            <span className="text-green-500 flex items-center text-sm">
              <FiArrowUp size={14} className="mr-1" />
              7.2%
            </span>
            <span className="text-gray-500 dark:text-gray-400 text-sm ml-2">vs. mes anterior</span>
          </div>
        </div>
        
        <div className="stat-card p-6 bg-white/10 dark:bg-primary-light/5 backdrop-blur-md rounded-xl border border-white/10 dark:border-primary-light/10 shadow-sm">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Ganancias</p>
              <h3 className="text-2xl font-bold text-primary-dark dark:text-white mt-1">
                ${dashboardData?.earnings.toLocaleString()}
              </h3>
            </div>
            <div className="p-2 bg-green-100 dark:bg-green-500/20 rounded-lg text-green-600 dark:text-green-400">
              <FiTrendingUp size={24} />
            </div>
          </div>
          <div className="mt-4 flex items-center">
            <span className="text-green-500 flex items-center text-sm">
              <FiArrowUp size={14} className="mr-1" />
              {dashboardData?.stats.monthly_growth}%
            </span>
            <span className="text-gray-500 dark:text-gray-400 text-sm ml-2">crecimiento mensual</span>
          </div>
        </div>
        
        <div className="stat-card p-6 bg-white/10 dark:bg-primary-light/5 backdrop-blur-md rounded-xl border border-white/10 dark:border-primary-light/10 shadow-sm">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Categoría</p>
              <h3 className="text-2xl font-bold text-primary-dark dark:text-white mt-1">
                {dashboardData?.category}
              </h3>
            </div>
            <div className="p-2 bg-yellow-100 dark:bg-yellow-500/20 rounded-lg text-yellow-600 dark:text-yellow-400">
              <FiBarChart size={24} />
            </div>
          </div>
          <div className="mt-4 flex items-center">
            <span className="text-blue-500 flex items-center text-sm">
              <FiClock size={14} className="mr-1" />
              {dashboardData?.stats.months_to_next_category}
            </span>
            <span className="text-gray-500 dark:text-gray-400 text-sm ml-2">meses para siguiente nivel</span>
          </div>
        </div>
        
        <div className="stat-card p-6 bg-white/10 dark:bg-primary-light/5 backdrop-blur-md rounded-xl border border-white/10 dark:border-primary-light/10 shadow-sm">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Bonos Acumulados</p>
              <h3 className="text-2xl font-bold text-primary-dark dark:text-white mt-1">
                ${dashboardData?.bonuses.toLocaleString()}
              </h3>
            </div>
            <div className="p-2 bg-purple-100 dark:bg-purple-500/20 rounded-lg text-purple-600 dark:text-purple-400">
              <FiActivity size={24} />
            </div>
          </div>
          <div className="mt-4 flex items-center">
            <span className="text-green-500 flex items-center text-sm">
              <FiArrowUp size={14} className="mr-1" />
              ${(dashboardData?.bonuses / 6).toFixed(2)}
            </span>
            <span className="text-gray-500 dark:text-gray-400 text-sm ml-2">promedio mensual</span>
          </div>
        </div>
      </div>
      
      {/* Gráficos */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="lg:col-span-2 chart-container p-6 bg-white/10 dark:bg-primary-light/5 backdrop-blur-md rounded-xl border border-white/10 dark:border-primary-light/10 shadow-sm">
          <h3 className="text-lg font-semibold text-primary-dark dark:text-white mb-4">
            Evolución de Capital y Ganancias
          </h3>
          <div className="h-80">
            <Line data={lineChartData} options={lineChartOptions} />
          </div>
        </div>
        
        <div className="chart-container p-6 bg-white/10 dark:bg-primary-light/5 backdrop-blur-md rounded-xl border border-white/10 dark:border-primary-light/10 shadow-sm">
          <h3 className="text-lg font-semibold text-primary-dark dark:text-white mb-4">
            Distribución de Portafolio
          </h3>
          <div className="h-80 flex items-center justify-center">
            <Doughnut data={doughnutChartData} options={doughnutChartOptions} />
          </div>
        </div>
      </div>
      
      {/* Transacciones recientes */}
      <div className="bg-white/10 dark:bg-primary-light/5 backdrop-blur-md rounded-xl border border-white/10 dark:border-primary-light/10 shadow-sm">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-primary-dark dark:text-white">
            Transacciones Recientes
          </h3>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-gray-500 dark:text-gray-400 text-sm">
                <th className="px-6 py-3 font-medium">ID</th>
                <th className="px-6 py-3 font-medium">Tipo</th>
                <th className="px-6 py-3 font-medium">Monto</th>
                <th className="px-6 py-3 font-medium">Fecha</th>
                <th className="px-6 py-3 font-medium">Estado</th>
              </tr>
            </thead>
            
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {dashboardData?.transactions.map((transaction, index) => (
                <tr 
                  key={transaction.id} 
                  className="transaction-item text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-primary-light/10 transition-colors"
                >
                  <td className="px-6 py-4 whitespace-nowrap font-medium">
                    {transaction.id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      transaction.type === 'deposit' 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' 
                        : transaction.type === 'profit'
                        ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
                        : transaction.type === 'bonus'
                        ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300'
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                    }`}>
                      {transaction.type === 'deposit' && 'Depósito'}
                      {transaction.type === 'profit' && 'Ganancia'}
                      {transaction.type === 'bonus' && 'Bono'}
                      {transaction.type === 'withdrawal' && 'Retiro'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`font-medium ${
                      transaction.type === 'withdrawal' 
                        ? 'text-red-500' 
                        : 'text-green-500'
                    }`}>
                      {transaction.type === 'withdrawal' ? '-' : '+'}${transaction.amount.toLocaleString()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {transaction.date}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      transaction.status === 'completed' 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' 
                        : transaction.status === 'pending'
                        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                        : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                    }`}>
                      {transaction.status === 'completed' && 'Completado'}
                      {transaction.status === 'pending' && 'Pendiente'}
                      {transaction.status === 'failed' && 'Fallido'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="p-4 text-center border-t border-gray-200 dark:border-gray-700">
          <button className="text-secondary hover:text-secondary-light transition-colors">
            Ver todas las transacciones
          </button>
        </div>
      </div>
    </MainLayout>
  );
};

export default Dashboard;