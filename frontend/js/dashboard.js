// Dashboard functionality
let dashboardData = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Set current date
    const currentDate = new Date().toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    document.getElementById('currentDate').textContent = currentDate;
    
    // Set user email
    const user = JSON.parse(localStorage.getItem('user'));
    if (user) {
        document.getElementById('userEmail').textContent = user.email;
    }
    
    // Load dashboard data
    loadDashboardData();
    
    // Load shelters for prediction
    loadShelters();
    
    // Set default date for prediction
    document.getElementById('predictionDate').value = new Date().toISOString().split('T')[0];
});

// Load dashboard data
async function loadDashboardData() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('/api/dashboard', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            dashboardData = await response.json();
            updateDashboard(dashboardData);
        } else {
            console.error('Failed to load dashboard data');
        }
    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}

// Update dashboard with data
function updateDashboard(data) {
    // Update stats
    document.getElementById('totalPredicted').textContent = data.total_predicted;
    document.getElementById('totalCapacity').textContent = data.shelters.length * 100; // Assuming 100 capacity per shelter
    document.getElementById('utilizationRate').textContent = Math.round((data.total_predicted / (data.shelters.length * 100)) * 100) + '%';
    document.getElementById('alertsCount').textContent = Math.floor(Math.random() * 5); // Random alerts for demo
    
    // Update shelter cards
    updateShelterCards(data.shelters);
    
    // Update resource stats
    updateResourceStats();
    
    // Update alerts
    updateAlerts();
}

// Update shelter cards
function updateShelterCards(shelters) {
    const container = document.getElementById('shelterCards');
    container.innerHTML = '';
    
    shelters.forEach(shelter => {
        const card = document.createElement('div');
        card.className = 'shelter-card';
        card.innerHTML = `
            <h3>${shelter.name}</h3>
            <div class="shelter-stats">
                <div class="shelter-stat">
                    <span class="number">${shelter.predicted_occupancy}</span>
                    <span class="label">Predicted</span>
                </div>
                <div class="shelter-stat">
                    <span class="number">${shelter.capacity}</span>
                    <span class="label">Capacity</span>
                </div>
                <div class="shelter-stat">
                    <span class="number">${shelter.utilization_rate}%</span>
                    <span class="label">Utilization</span>
                </div>
            </div>
            <div class="utilization-bar">
                <div class="utilization-fill" style="width: ${shelter.utilization_rate}%"></div>
            </div>
        `;
        container.appendChild(card);
    });
}

// Update resource stats
function updateResourceStats() {
    // Simulate resource data
    document.getElementById('totalBeds').textContent = '700';
    document.getElementById('allocatedBeds').textContent = '450';
    document.getElementById('totalStaff').textContent = '120';
    document.getElementById('onDutyStaff').textContent = '85';
    document.getElementById('foodSupplies').textContent = '500';
    document.getElementById('medicalSupplies').textContent = '50';
}

// Update alerts
function updateAlerts() {
    const alertsList = document.getElementById('alertsList');
    alertsList.innerHTML = '';
    
    const alerts = [
        {
            type: 'critical',
            icon: 'fas fa-exclamation-triangle',
            title: 'High Occupancy Alert',
            message: 'Christie Ossington Men\'s Hostel is at 95% capacity'
        },
        {
            type: 'warning',
            icon: 'fas fa-clock',
            title: 'Staff Shortage',
            message: 'Birkdale Residence needs additional staff for tonight'
        },
        {
            type: 'info',
            icon: 'fas fa-info-circle',
            title: 'Supply Update',
            message: 'Medical supplies restocked at COSTI Reception Centre'
        }
    ];
    
    alerts.forEach(alert => {
        const alertItem = document.createElement('div');
        alertItem.className = `alert-item ${alert.type}`;
        alertItem.innerHTML = `
            <i class="${alert.icon} alert-icon"></i>
            <div class="alert-content">
                <h4>${alert.title}</h4>
                <p>${alert.message}</p>
            </div>
        `;
        alertsList.appendChild(alertItem);
    });
}

// Load shelters for prediction dropdown
async function loadShelters() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('/api/shelters', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            const select = document.getElementById('predictionShelter');
            
            data.shelters.forEach(shelter => {
                const option = document.createElement('option');
                option.value = shelter;
                option.textContent = shelter;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading shelters:', error);
    }
}

// Make prediction
async function makePrediction() {
    const date = document.getElementById('predictionDate').value;
    const shelter = document.getElementById('predictionShelter').value;
    
    if (!date || !shelter) {
        alert('Please select both date and shelter');
        return;
    }
    
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ date, shelter_name: shelter })
        });
        
        if (response.ok) {
            const data = await response.json();
            showPredictionResult(data);
        } else {
            alert('Failed to make prediction');
        }
    } catch (error) {
        console.error('Error making prediction:', error);
        alert('Error making prediction');
    }
}

// Show prediction result
function showPredictionResult(data) {
    document.getElementById('predictionShelterName').textContent = data.shelter_name;
    document.getElementById('predictionDateDisplay').textContent = new Date(data.date).toLocaleDateString();
    document.getElementById('predictionValue').textContent = data.predicted_occupancy;
    document.getElementById('predictionResult').style.display = 'block';
}

// Show different sections
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.dashboard-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Remove active class from all nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Show selected section
    document.getElementById(sectionId).classList.add('active');
    
    // Add active class to clicked nav item
    event.target.classList.add('active');
}

// Search and filter shelters
document.getElementById('shelterSearch').addEventListener('input', function() {
    const searchTerm = this.value.toLowerCase();
    const cards = document.querySelectorAll('.shelter-card');
    
    cards.forEach(card => {
        const shelterName = card.querySelector('h3').textContent.toLowerCase();
        if (shelterName.includes(searchTerm)) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
});

document.getElementById('shelterFilter').addEventListener('change', function() {
    const filterValue = this.value;
    const cards = document.querySelectorAll('.shelter-card');
    
    cards.forEach(card => {
        const utilization = parseInt(card.querySelector('.shelter-stat:last-child .number').textContent);
        
        if (filterValue === 'high' && utilization >= 80) {
            card.style.display = 'block';
        } else if (filterValue === 'low' && utilization < 50) {
            card.style.display = 'block';
        } else if (filterValue === '') {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}); 