// Authentication functionality
let currentUser = null;

// Check if user is logged in
function checkAuth() {
    const token = localStorage.getItem('token');
    if (token) {
        currentUser = JSON.parse(localStorage.getItem('user'));
        if (window.location.pathname === '/') {
            window.location.href = '/dashboard';
        }
    } else {
        if (window.location.pathname === '/dashboard') {
            window.location.href = '/';
        }
    }
}

// Show login modal
function showLoginModal() {
    document.getElementById('loginModal').style.display = 'block';
}

// Show register modal
function showRegisterModal() {
    document.getElementById('registerModal').style.display = 'block';
}

// Close modal
function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

// Switch between login and register
function switchToRegister() {
    closeModal('loginModal');
    showRegisterModal();
}

function switchToLogin() {
    closeModal('registerModal');
    showLoginModal();
}

// Handle login form submission
document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            localStorage.setItem('token', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));
            currentUser = data.user;
            
            closeModal('loginModal');
            window.location.href = '/dashboard';
        } else {
            alert(data.error || 'Login failed');
        }
    } catch (error) {
        console.error('Login error:', error);
        alert('Login failed. Please try again.');
    }
});

// Handle register form submission
document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = document.getElementById('registerUsername').value;
    const email = document.getElementById('registerEmail').value;
    const password = document.getElementById('registerPassword').value;
    
    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            localStorage.setItem('token', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));
            currentUser = data.user;
            
            closeModal('registerModal');
            window.location.href = '/dashboard';
        } else {
            alert(data.error || 'Registration failed');
        }
    } catch (error) {
        console.error('Registration error:', error);
        alert('Registration failed. Please try again.');
    }
});

// Logout function
function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    currentUser = null;
    window.location.href = '/';
}

// Close modals when clicking outside
window.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.style.display = 'none';
    }
});

// Check auth on page load
document.addEventListener('DOMContentLoaded', checkAuth); 