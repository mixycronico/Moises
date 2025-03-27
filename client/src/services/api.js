import axios from 'axios';

// Crear instancia de axios con configuración base
const api = axios.create({
  baseURL: '/api', // Usamos URL relativa para evitar problemas de CORS
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  }
});

// Interceptor para manejar tokens en las solicitudes
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Interceptor para manejar respuestas y errores globalmente
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    // Si es error 401 (no autorizado), podemos redirigir al login
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Servicios API

// Auth
export const checkAuthStatus = () => api.get('/auth/status');
export const login = (credentials) => api.post('/auth/login', credentials);
export const logout = () => api.post('/auth/logout');

// Dashboard
export const getInvestorDashboard = () => api.get('/investor/dashboard');

// Notificaciones
export const getNotifications = () => api.get('/notifications');

// Bonos
export const getBonusStatus = () => api.get('/bonus/status');
export const simulateBonus = () => api.post('/bonus/simulate');

// Préstamos
export const getLoanStatus = () => api.get('/loan/status');
export const requestLoan = (amount) => api.post('/loan/request', { amount });
export const makeLoanPayment = (loanId, amount) => api.post(`/loan/payment/${loanId}`, { amount });

export default api;