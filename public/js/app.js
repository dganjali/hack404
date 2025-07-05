// Utility functions
function showToast(message, isError = false) {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.style.background = isError ? '#ef4444' : '#2563eb';
  toast.style.display = 'block';
  setTimeout(() => { toast.style.display = 'none'; }, 3000);
}

function showModal(id) {
  document.getElementById(id).style.display = 'flex';
}
function closeModal(id) {
  document.getElementById(id).style.display = 'none';
}

function setAuthState(isLoggedIn, user = null) {
  document.getElementById('dashboard').style.display = isLoggedIn ? 'block' : 'none';
  document.getElementById('welcome').style.display = isLoggedIn ? 'none' : 'block';
  document.getElementById('loginBtn').style.display = isLoggedIn ? 'none' : 'inline-block';
  document.getElementById('registerBtn').style.display = isLoggedIn ? 'none' : 'inline-block';
  document.getElementById('logoutBtn').style.display = isLoggedIn ? 'inline-block' : 'none';
  if (isLoggedIn && user) {
    document.getElementById('userName').textContent = user.username;
  }
}

function getToken() {
  return localStorage.getItem('token');
}
function setToken(token) {
  localStorage.setItem('token', token);
}
function clearToken() {
  localStorage.removeItem('token');
}

// API base
const API_BASE = '/api';

// Auth
async function login(email, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });
  if (!res.ok) throw new Error((await res.json()).error || 'Login failed');
  return res.json();
}
async function register(username, email, password, organization) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password, organization })
  });
  if (!res.ok) throw new Error((await res.json()).error || 'Registration failed');
  return res.json();
}
async function getMe() {
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: { 'Authorization': 'Bearer ' + getToken() }
  });
  if (!res.ok) throw new Error('Not authenticated');
  return res.json();
}

// Dashboard data
async function fetchOverview() {
  const res = await fetch(`${API_BASE}/shelters/stats/overview`, {
    headers: { 'Authorization': 'Bearer ' + getToken() }
  });
  if (!res.ok) throw new Error('Failed to fetch overview');
  return (await res.json()).overview;
}
async function fetchShelters() {
  const res = await fetch(`${API_BASE}/shelters`, {
    headers: { 'Authorization': 'Bearer ' + getToken() }
  });
  if (!res.ok) throw new Error('Failed to fetch shelters');
  return (await res.json()).shelters;
}
async function fetchTrends(days = 30) {
  const res = await fetch(`${API_BASE}/shelters/shelter_1874/trends?days=${days}`, {
    headers: { 'Authorization': 'Bearer ' + getToken() }
  });
  if (!res.ok) throw new Error('Failed to fetch trends');
  return (await res.json()).trends;
}

// Chart.js instance
let occupancyChart = null;
function renderOccupancyChart(trends) {
  const ctx = document.getElementById('occupancyChart').getContext('2d');
  if (occupancyChart) occupancyChart.destroy();
  occupancyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: trends.dates,
      datasets: [{
        label: 'Total Occupancy',
        data: trends.totalOccupancy,
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37,99,235,0.1)',
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { display: true }, y: { display: true } }
    }
  });
}

// Populate dashboard
async function loadDashboard() {
  try {
    const [overview, shelters, trends] = await Promise.all([
      fetchOverview(),
      fetchShelters(),
      fetchTrends(30)
    ]);
    document.getElementById('totalShelters').textContent = overview.totalShelters;
    document.getElementById('totalOccupancy').textContent = overview.totalOccupancy;
    document.getElementById('avgUtilization').textContent = (overview.avgUtilization * 100).toFixed(1) + '%';
    document.getElementById('dateRange').textContent = `${overview.dateRange.start} - ${overview.dateRange.end}`;
    // Top shelters
    const topSheltersBody = document.querySelector('#topSheltersTable tbody');
    topSheltersBody.innerHTML = '';
    overview.topShelters.forEach(s => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${s.name}</td><td>${(s.utilization * 100).toFixed(1)}%</td><td>${s.avgOccupancy.toFixed(1)}</td><td>${s.capacity}</td>`;
      topSheltersBody.appendChild(tr);
    });
    // All shelters
    const sheltersBody = document.querySelector('#sheltersTable tbody');
    sheltersBody.innerHTML = '';
    shelters.forEach(s => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${s.name}</td><td>${s.sector}</td><td>${s.capacity}</td><td>${(s.utilization_rate * 100).toFixed(1)}%</td><td><button class="btn btn-secondary" onclick="alert('Details coming soon!')">View</button></td>`;
      sheltersBody.appendChild(tr);
    });
    // Chart
    renderOccupancyChart(trends);
  } catch (e) {
    showToast('Failed to load dashboard', true);
  }
}

// Auth UI
function setupAuthUI() {
  // Modals
  const loginModal = document.getElementById('loginModal');
  const registerModal = document.getElementById('registerModal');
  document.getElementById('loginBtn').onclick = () => showModal('loginModal');
  document.getElementById('registerBtn').onclick = () => showModal('registerModal');
  document.getElementById('closeLogin').onclick = () => closeModal('loginModal');
  document.getElementById('closeRegister').onclick = () => closeModal('registerModal');
  document.getElementById('switchToRegister').onclick = (e) => { e.preventDefault(); closeModal('loginModal'); showModal('registerModal'); };
  document.getElementById('switchToLogin').onclick = (e) => { e.preventDefault(); closeModal('registerModal'); showModal('loginModal'); };
  // Logout
  document.getElementById('logoutBtn').onclick = () => {
    clearToken();
    setAuthState(false);
    showToast('Logged out');
  };
  // Login form
  document.getElementById('loginForm').onsubmit = async (e) => {
    e.preventDefault();
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    try {
      const { token, user } = await login(email, password);
      setToken(token);
      setAuthState(true, user);
      closeModal('loginModal');
      showToast('Login successful');
      loadDashboard();
    } catch (err) {
      showToast(err.message, true);
    }
  };
  // Register form
  document.getElementById('registerForm').onsubmit = async (e) => {
    e.preventDefault();
    const username = document.getElementById('registerUsername').value;
    const email = document.getElementById('registerEmail').value;
    const password = document.getElementById('registerPassword').value;
    const organization = document.getElementById('registerOrganization').value;
    try {
      const { token, user } = await register(username, email, password, organization);
      setToken(token);
      setAuthState(true, user);
      closeModal('registerModal');
      showToast('Registration successful');
      loadDashboard();
    } catch (err) {
      showToast(err.message, true);
    }
  };
}

// On load
window.onload = async function() {
  setupAuthUI();
  // Try auto-login
  if (getToken()) {
    try {
      const { user } = await getMe();
      setAuthState(true, user);
      loadDashboard();
    } catch {
      setAuthState(false);
    }
  } else {
    setAuthState(false);
  }
}; 