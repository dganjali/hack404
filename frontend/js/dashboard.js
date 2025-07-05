// Dashboard JavaScript
let currentUser = null;
let userShelters = [];
let map = null;
let markers = [];

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    checkAuth();
    setCurrentDate();
    loadDashboardData();
    setupEventListeners();
});

// Check authentication
function checkAuth() {
    const token = localStorage.getItem('token');
    if (!token) {
        window.location.href = '/';
        return;
    }
    
    currentUser = JSON.parse(localStorage.getItem('user'));
    document.getElementById('userEmail').textContent = currentUser.email;
}

// Set current date
function setCurrentDate() {
    const today = new Date();
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    document.getElementById('currentDate').textContent = today.toLocaleDateString('en-US', options);
}

// Setup event listeners
function setupEventListeners() {
    // Add shelter form
    document.getElementById('addShelterForm').addEventListener('submit', handleAddShelter);
    
    // Edit shelter form
    document.getElementById('editShelterForm').addEventListener('submit', handleEditShelter);
    
    // Data entry form
    document.getElementById('dataEntryForm').addEventListener('submit', handleDataEntry);
    
    // Search and filter
    document.getElementById('shelterSearch').addEventListener('input', filterShelters);
    document.getElementById('shelterFilter').addEventListener('change', filterShelters);
}

// Load dashboard data
async function loadDashboardData() {
    try {
        const response = await fetch('/api/dashboard', {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load dashboard data');
        }
        
        const data = await response.json();
        updateDashboardStats(data);
        updateShelterCards(data.shelters);
        updateShelterTable(data.shelters);
        updateDataEntryForm(data.shelters);
        
        // Show welcome message if no shelters
        if (data.shelters.length === 0) {
            document.getElementById('welcomeMessage').style.display = 'block';
            document.getElementById('shelterList').style.display = 'none';
        } else {
            document.getElementById('welcomeMessage').style.display = 'none';
            document.getElementById('shelterList').style.display = 'block';
        }
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showNotification('Error loading dashboard data', 'error');
    }
}

// Update dashboard statistics
function updateDashboardStats(data) {
    document.getElementById('totalPredicted').textContent = data.total_predicted;
    document.getElementById('totalCapacity').textContent = data.total_capacity;
    document.getElementById('utilizationRate').textContent = data.utilization_rate + '%';
    document.getElementById('alertsCount').textContent = data.alerts_count;
}

// Update shelter cards
function updateShelterCards(shelters) {
    const container = document.getElementById('shelterCards');
    container.innerHTML = '';
    
    shelters.forEach(shelter => {
        const card = createShelterCard(shelter);
        container.appendChild(card);
    });
}

// Create shelter card
function createShelterCard(shelter) {
    const card = document.createElement('div');
    card.className = 'shelter-card';
    
    const utilizationClass = getUtilizationClass(shelter.utilization_rate);
    
    card.innerHTML = `
        <div class="shelter-card-header">
            <h3 class="shelter-card-title">${shelter.name}</h3>
            <div class="shelter-card-actions">
                <button class="btn btn-sm btn-outline" onclick="editShelter('${shelter.id}')">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="deleteShelter('${shelter.id}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
        <div class="shelter-card-info">
            <div class="shelter-info-item">
                <span class="shelter-info-label">Address</span>
                <span class="shelter-info-value">${shelter.address}</span>
            </div>
            <div class="shelter-info-item">
                <span class="shelter-info-label">Sector</span>
                <span class="shelter-info-value">${shelter.sector_info ? shelter.sector_info.sector_name : 'Unknown'}</span>
            </div>
            <div class="shelter-info-item">
                <span class="shelter-info-label">Capacity</span>
                <span class="shelter-info-value">${shelter.maxCapacity}</span>
            </div>
            <div class="shelter-info-item">
                <span class="shelter-info-label">Current</span>
                <span class="shelter-info-value">${shelter.currentOccupancy}</span>
            </div>
            <div class="shelter-info-item">
                <span class="shelter-info-label">Predicted</span>
                <span class="shelter-info-value">${shelter.predicted_occupancy}</span>
            </div>
        </div>
        <div class="shelter-utilization ${utilizationClass}">
            <strong>${shelter.utilization_rate}%</strong> Utilization
        </div>
    `;
    
    return card;
}

// Get utilization class
function getUtilizationClass(rate) {
    if (rate >= 80) return 'utilization-high';
    if (rate >= 60) return 'utilization-medium';
    return 'utilization-low';
}

// Update shelter table
function updateShelterTable(shelters) {
    const container = document.getElementById('shelterTable');
    container.innerHTML = '';
    
    if (shelters.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-building"></i>
                <h3>No Shelters</h3>
                <p>Add your first shelter to get started</p>
            </div>
        `;
        return;
    }
    
    const table = document.createElement('table');
    table.innerHTML = `
        <thead>
            <tr>
                <th>Name</th>
                <th>Address</th>
                <th>Sector</th>
                <th>Capacity</th>
                <th>Current</th>
                <th>Predicted</th>
                <th>Utilization</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            ${shelters.map(shelter => `
                <tr>
                    <td>${shelter.name}</td>
                    <td>${shelter.address}</td>
                    <td>${shelter.sector_info ? shelter.sector_info.sector_name : 'Unknown'}</td>
                    <td>${shelter.maxCapacity}</td>
                    <td>${shelter.currentOccupancy}</td>
                    <td>${shelter.predicted_occupancy}</td>
                    <td>
                        <span class="utilization-badge ${getUtilizationClass(shelter.utilization_rate)}">
                            ${shelter.utilization_rate}%
                        </span>
                    </td>
                    <td>
                        <button class="btn btn-sm btn-outline" onclick="editShelter('${shelter.id}')">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-sm btn-danger" onclick="deleteShelter('${shelter.id}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `).join('')}
        </tbody>
    `;
    
    container.appendChild(table);
}

// Update data entry form
function updateDataEntryForm(shelters) {
    const select = document.getElementById('dataShelter');
    select.innerHTML = '<option value="">Select a shelter</option>';
    
    shelters.forEach(shelter => {
        const option = document.createElement('option');
        option.value = shelter.id;
        option.textContent = shelter.name;
        select.appendChild(option);
    });
}

// Show section
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
    
    // Add active class to nav item
    event.target.classList.add('active');
    
    // Load section-specific data
    if (sectionId === 'alerts') {
        loadAlerts();
    } else if (sectionId === 'map') {
        loadMap();
    }
}

// Load alerts
async function loadAlerts() {
    try {
        const response = await fetch('/api/alerts', {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load alerts');
        }
        
        const data = await response.json();
        updateAlertsList(data.alerts);
        
    } catch (error) {
        console.error('Error loading alerts:', error);
        showNotification('Error loading alerts', 'error');
    }
}

// Update alerts list
function updateAlertsList(alerts) {
    const container = document.getElementById('alertsList');
    container.innerHTML = '';
    
    if (alerts.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-bell"></i>
                <h3>No Alerts</h3>
                <p>You're all caught up!</p>
            </div>
        `;
        return;
    }
    
    alerts.forEach(alert => {
        const alertElement = createAlertElement(alert);
        container.appendChild(alertElement);
    });
}

// Create alert element
function createAlertElement(alert) {
    const div = document.createElement('div');
    div.className = `alert-item ${alert.severity}`;
    
    div.innerHTML = `
        <div class="alert-content">
            <div class="alert-title">${alert.title}</div>
            <div class="alert-message">${alert.message}</div>
            <div class="alert-meta">
                ${new Date(alert.timestamp).toLocaleString()}
            </div>
        </div>
        <div class="alert-actions">
            ${!alert.read ? `
                <button class="btn btn-sm btn-outline" onclick="markAlertRead('${alert.id}')">
                    Mark Read
                </button>
            ` : ''}
        </div>
    `;
    
    return div;
}

// Mark alert as read
async function markAlertRead(alertId) {
    try {
        const response = await fetch(`/api/alerts/${alertId}/read`, {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to mark alert as read');
        }
        
        loadAlerts();
        loadDashboardData(); // Refresh dashboard stats
        
    } catch (error) {
        console.error('Error marking alert as read:', error);
        showNotification('Error marking alert as read', 'error');
    }
}

// Load map
function loadMap() {
    const container = document.getElementById('mapContainer');
    
    if (!map) {
        // Initialize map
        map = new google.maps.Map(container, {
            center: { lat: 43.6532, lng: -79.3832 }, // Toronto
            zoom: 10
        });
    }
    
    // Clear existing markers
    markers.forEach(marker => marker.setMap(null));
    markers = [];
    
    // Add markers for each shelter
    userShelters.forEach(shelter => {
        if (shelter.address) {
            geocodeAddress(shelter);
        }
    });
}

// Geocode address and add marker
function geocodeAddress(shelter) {
    const geocoder = new google.maps.Geocoder();
    
    geocoder.geocode({ address: shelter.address }, (results, status) => {
        if (status === 'OK') {
            const location = results[0].geometry.location;
            
            const marker = new google.maps.Marker({
                map: map,
                position: location,
                title: shelter.name
            });
            
            const infoWindow = new google.maps.InfoWindow({
                content: `
                    <div style="padding: 10px;">
                        <h3>${shelter.name}</h3>
                        <p><strong>Address:</strong> ${shelter.address}</p>
                        <p><strong>Capacity:</strong> ${shelter.maxCapacity}</p>
                        <p><strong>Phone:</strong> ${shelter.phone || 'N/A'}</p>
                    </div>
                `
            });
            
            marker.addListener('click', () => {
                infoWindow.open(map, marker);
            });
            
            markers.push(marker);
        }
    });
}

// Show add shelter modal
function showAddShelterModal() {
    document.getElementById('addShelterModal').style.display = 'block';
    document.getElementById('addShelterForm').reset();
}

// Close add shelter modal
function closeAddShelterModal() {
    document.getElementById('addShelterModal').style.display = 'none';
}

// Handle add shelter
async function handleAddShelter(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const shelterData = {
        name: formData.get('shelterName') || document.getElementById('shelterName').value,
        address: formData.get('shelterAddress') || document.getElementById('shelterAddress').value,
        maxCapacity: formData.get('shelterMaxCapacity') || document.getElementById('shelterMaxCapacity').value,
        phone: formData.get('shelterPhone') || document.getElementById('shelterPhone').value,
        email: formData.get('shelterEmail') || document.getElementById('shelterEmail').value,
        description: formData.get('shelterDescription') || document.getElementById('shelterDescription').value,
        postal_code: formData.get('shelterPostalCode') || document.getElementById('shelterPostalCode').value
    };
    
    try {
        const response = await fetch('/api/shelters', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify(shelterData)
        });
        
        if (!response.ok) {
            throw new Error('Failed to add shelter');
        }
        
        closeAddShelterModal();
        loadDashboardData();
        showNotification('Shelter added successfully', 'success');
        
    } catch (error) {
        console.error('Error adding shelter:', error);
        showNotification('Error adding shelter', 'error');
    }
}

// Edit shelter
async function editShelter(shelterId) {
    try {
        const response = await fetch(`/api/shelters/${shelterId}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to get shelter details');
        }
        
        const shelter = await response.json();
        
        // Populate edit form
        document.getElementById('editShelterId').value = shelter.id;
        document.getElementById('editShelterName').value = shelter.name;
        document.getElementById('editShelterAddress').value = shelter.address;
        document.getElementById('editShelterMaxCapacity').value = shelter.maxCapacity;
        document.getElementById('editShelterPhone').value = shelter.phone || '';
        document.getElementById('editShelterEmail').value = shelter.email || '';
        document.getElementById('editShelterDescription').value = shelter.description || '';
        document.getElementById('editShelterPostalCode').value = shelter.postal_code || '';
        
        // Show edit modal
        document.getElementById('editShelterModal').style.display = 'block';
        
    } catch (error) {
        console.error('Error getting shelter details:', error);
        showNotification('Error loading shelter details', 'error');
    }
}

// Close edit shelter modal
function closeEditShelterModal() {
    document.getElementById('editShelterModal').style.display = 'none';
}

// Handle edit shelter
async function handleEditShelter(event) {
    event.preventDefault();
    
    const shelterId = document.getElementById('editShelterId').value;
    const shelterData = {
        name: document.getElementById('editShelterName').value,
        address: document.getElementById('editShelterAddress').value,
        maxCapacity: document.getElementById('editShelterMaxCapacity').value,
        phone: document.getElementById('editShelterPhone').value,
        email: document.getElementById('editShelterEmail').value,
        description: document.getElementById('editShelterDescription').value,
        postal_code: document.getElementById('editShelterPostalCode').value
    };
    
    try {
        const response = await fetch(`/api/shelters/${shelterId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify(shelterData)
        });
        
        if (!response.ok) {
            throw new Error('Failed to update shelter');
        }
        
        closeEditShelterModal();
        loadDashboardData();
        showNotification('Shelter updated successfully', 'success');
        
    } catch (error) {
        console.error('Error updating shelter:', error);
        showNotification('Error updating shelter', 'error');
    }
}

// Delete shelter
async function deleteShelter(shelterId) {
    if (!confirm('Are you sure you want to delete this shelter? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/shelters/${shelterId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to delete shelter');
        }
        
        loadDashboardData();
        showNotification('Shelter deleted successfully', 'success');
        
    } catch (error) {
        console.error('Error deleting shelter:', error);
        showNotification('Error deleting shelter', 'error');
    }
}

// Handle data entry
async function handleDataEntry(event) {
    event.preventDefault();
    
    const shelterId = document.getElementById('dataShelter').value;
    const date = document.getElementById('dataDate').value;
    const occupancy = document.getElementById('dataOccupancy').value;
    const notes = document.getElementById('dataNotes').value;
    
    if (!shelterId || !date || !occupancy) {
        showNotification('Please fill in all required fields', 'error');
        return;
    }
    
    try {
        const response = await fetch(`/api/shelters/${shelterId}/data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                date,
                occupancy: parseInt(occupancy),
                notes
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to add data');
        }
        
        event.target.reset();
        loadDashboardData();
        loadDataHistory();
        showNotification('Data added successfully', 'success');
        
    } catch (error) {
        console.error('Error adding data:', error);
        showNotification('Error adding data', 'error');
    }
}

// Load data history
async function loadDataHistory() {
    try {
        const shelterId = document.getElementById('dataShelter').value;
        if (!shelterId) return;
        
        const response = await fetch(`/api/shelters/${shelterId}/data`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load data history');
        }
        
        const data = await response.json();
        updateDataHistory(data.data);
        
    } catch (error) {
        console.error('Error loading data history:', error);
    }
}

// Update data history
function updateDataHistory(dataPoints) {
    const container = document.getElementById('dataHistory');
    container.innerHTML = '';
    
    if (dataPoints.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-chart-line"></i>
                <h3>No Data</h3>
                <p>No occupancy data has been entered yet</p>
            </div>
        `;
        return;
    }
    
    // Sort by date (newest first)
    dataPoints.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    dataPoints.slice(0, 10).forEach(dataPoint => {
        const div = document.createElement('div');
        div.className = 'data-history-item';
        
        div.innerHTML = `
            <div class="data-history-info">
                <div class="data-history-shelter">${dataPoint.shelterName || 'Shelter'}</div>
                <div class="data-history-details">
                    ${new Date(dataPoint.date).toLocaleDateString()}
                    ${dataPoint.notes ? ` - ${dataPoint.notes}` : ''}
                </div>
            </div>
            <div class="data-history-occupancy">
                ${dataPoint.occupancy}
            </div>
        `;
        
        container.appendChild(div);
    });
}

// Filter shelters
function filterShelters() {
    const searchTerm = document.getElementById('shelterSearch').value.toLowerCase();
    const filterValue = document.getElementById('shelterFilter').value;
    
    // This would be implemented with the actual shelter data
    // For now, we'll just reload the dashboard data
    loadDashboardData();
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 6px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    
    if (type === 'success') {
        notification.style.backgroundColor = '#27ae60';
    } else if (type === 'error') {
        notification.style.backgroundColor = '#e74c3c';
    } else {
        notification.style.backgroundColor = '#3498db';
    }
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .utilization-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .utilization-badge.utilization-high {
        background-color: #ffe8e8;
        color: #d63031;
    }
    
    .utilization-badge.utilization-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .utilization-badge.utilization-low {
        background-color: #e8f5e8;
        color: #27ae60;
    }
`;
document.head.appendChild(style);

// Logout function
function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/';
} 