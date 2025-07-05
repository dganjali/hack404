#!/usr/bin/env python3
"""
Mock data generator for NeedsMatcher
Generates realistic shelter and intake history data
"""

import json
import random
from datetime import datetime, timedelta
import os

def generate_shelters():
    """Generate mock shelter data"""
    shelters = [
        {
            "id": "shelter_1",
            "name": "Downtown Hope Center",
            "capacity": 50,
            "current_beds": 35,
            "current_meals": 120,
            "current_kits": 45
        },
        {
            "id": "shelter_2",
            "name": "Northside Refuge",
            "capacity": 40,
            "current_beds": 28,
            "current_meals": 95,
            "current_kits": 32
        },
        {
            "id": "shelter_3",
            "name": "East End Shelter",
            "capacity": 35,
            "current_beds": 22,
            "current_meals": 80,
            "current_kits": 28
        },
        {
            "id": "shelter_4",
            "name": "Westside Safe Haven",
            "capacity": 45,
            "current_beds": 38,
            "current_meals": 110,
            "current_kits": 40
        },
        {
            "id": "shelter_5",
            "name": "Central Community Shelter",
            "capacity": 60,
            "current_beds": 45,
            "current_meals": 140,
            "current_kits": 55
        }
    ]
    return shelters

def generate_intake_history(shelters, days=30):
    """Generate realistic intake history data"""
    history = []
    start_date = datetime.now() - timedelta(days=days)
    
    for shelter in shelters:
        # Base demand for each shelter
        base_beds = random.randint(15, 40)
        base_meals = int(base_beds * random.uniform(1.5, 2.2))
        base_kits = int(base_beds * random.uniform(0.8, 1.2))
        
        for day in range(days):
            date = start_date + timedelta(days=day)
            
            # Add weekly patterns (weekends have higher demand)
            weekend_factor = 1.2 if date.weekday() >= 5 else 1.0
            
            # Add some random variation
            variation = random.uniform(0.8, 1.2)
            
            # Add trend (slight increase over time)
            trend_factor = 1.0 + (day / days) * 0.1
            
            beds_needed = int(base_beds * weekend_factor * variation * trend_factor)
            meals_needed = int(base_meals * weekend_factor * variation * trend_factor)
            kits_needed = int(base_kits * weekend_factor * variation * trend_factor)
            
            history.append({
                "shelter_id": shelter["id"],
                "date": date.strftime("%Y-%m-%d"),
                "beds_needed": max(0, beds_needed),
                "meals_needed": max(0, meals_needed),
                "kits_needed": max(0, kits_needed)
            })
    
    return history

def main():
    """Generate and save mock data"""
    print("Generating mock data for NeedsMatcher...")
    
    # Create data directory if it doesn't exist
    os.makedirs("../backend", exist_ok=True)
    
    # Generate data
    shelters = generate_shelters()
    history = generate_intake_history(shelters, days=30)
    
    # Save shelters data
    with open("../backend/shelters.json", "w") as f:
        json.dump(shelters, f, indent=2)
    print(f"✅ Generated {len(shelters)} shelters")
    
    # Save intake history
    with open("../backend/intake_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"✅ Generated {len(history)} intake records")
    
    # Print summary
    print("\n📊 Data Summary:")
    print(f"   • {len(shelters)} shelters")
    print(f"   • {len(history)} daily intake records")
    print(f"   • {len(history) // len(shelters)} days per shelter")
    
    print("\n🎯 Next Steps:")
    print("   1. Start the backend: cd ../backend && uvicorn main:app --reload")
    print("   2. Start the frontend: cd ../frontend && npm start")
    print("   3. Open http://localhost:3000 to view the dashboard")

if __name__ == "__main__":
    main() 