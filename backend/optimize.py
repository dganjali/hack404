import pulp
from typing import List, Dict, Tuple
import numpy as np

class ResourceOptimizer:
    def __init__(self):
        self.transfer_cost = 1.0  # Cost per unit transferred
        self.shortage_penalty = 10.0  # Penalty for shortages
        
    def optimize(self, shelters: List[Dict], forecasts: Dict[str, Dict]) -> Tuple[List[Dict], float]:
        """
        Optimize resource allocation between shelters using linear programming
        
        Args:
            shelters: List of shelter data with current inventory
            forecasts: Dictionary of predicted needs per shelter
            
        Returns:
            Tuple of (transfers, shortage_reduction_percentage)
        """
        if not shelters or not forecasts:
            return [], 0.0
        
        # Create optimization problem
        prob = pulp.LpProblem("Shelter_Resource_Optimization", pulp.LpMinimize)
        
        # Decision variables: transfers between shelters
        transfers = {}
        items = ['beds', 'meals', 'kits']
        
        for i, shelter_from in enumerate(shelters):
            for j, shelter_to in enumerate(shelters):
                if i != j:  # No self-transfers
                    for item in items:
                        var_name = f"transfer_{shelter_from['id']}_{shelter_to['id']}_{item}"
                        transfers[var_name] = pulp.LpVariable(var_name, 0, None)
        
        # Objective function: minimize total cost
        objective = 0
        for var in transfers.values():
            objective += self.transfer_cost * var
        prob += objective
        
        # Constraints
        
        # 1. Cannot transfer more than available inventory
        for shelter in shelters:
            shelter_id = shelter['id']
            if shelter_id not in forecasts:
                continue
                
            for item in items:
                current_inventory = shelter[f'current_{item}']
                constraint = 0
                
                # Sum of all transfers out
                for other_shelter in shelters:
                    if other_shelter['id'] != shelter_id:
                        var_name = f"transfer_{shelter_id}_{other_shelter['id']}_{item}"
                        if var_name in transfers:
                            constraint += transfers[var_name]
                
                prob += constraint <= current_inventory, f"inventory_limit_{shelter_id}_{item}"
        
        # 2. Balance constraints (optional - can be relaxed)
        # This ensures conservation of resources across the system
        
        # 3. Minimize shortages
        shortage_vars = {}
        for shelter in shelters:
            shelter_id = shelter['id']
            if shelter_id not in forecasts:
                continue
                
            for item in items:
                predicted_need = forecasts[shelter_id][item]
                current_inventory = shelter[f'current_{item}']
                
                # Calculate net transfers in/out
                transfers_in = 0
                transfers_out = 0
                
                for other_shelter in shelters:
                    if other_shelter['id'] != shelter_id:
                        # Transfers in
                        var_name_in = f"transfer_{other_shelter['id']}_{shelter_id}_{item}"
                        if var_name_in in transfers:
                            transfers_in += transfers[var_name_in]
                        
                        # Transfers out
                        var_name_out = f"transfer_{shelter_id}_{other_shelter['id']}_{item}"
                        if var_name_out in transfers:
                            transfers_out += transfers[var_name_out]
                
                # Shortage variable
                shortage_var_name = f"shortage_{shelter_id}_{item}"
                shortage_vars[shortage_var_name] = pulp.LpVariable(shortage_var_name, 0, None)
                
                # Constraint: shortage = max(0, predicted_need - (current + transfers_in - transfers_out))
                prob += shortage_vars[shortage_var_name] >= predicted_need - (current_inventory + transfers_in - transfers_out), f"shortage_def_{shelter_id}_{item}"
                prob += shortage_vars[shortage_var_name] >= 0, f"shortage_nonneg_{shelter_id}_{item}"
        
        # Add shortage penalty to objective
        for shortage_var in shortage_vars.values():
            objective += self.shortage_penalty * shortage_var
        
        # Solve the problem
        prob.solve()
        
        if prob.status != pulp.LpStatusOptimal:
            print(f"Warning: Optimization status: {pulp.LpStatus[prob.status]}")
            return [], 0.0
        
        # Extract results
        transfers_list = []
        for var_name, var in transfers.items():
            if var.varValue > 0:
                # Parse variable name: transfer_from_to_item
                parts = var_name.split('_')
                from_shelter = parts[1]
                to_shelter = parts[2]
                item = parts[3]
                
                transfers_list.append({
                    "from": from_shelter,
                    "to": to_shelter,
                    "item": item,
                    "amount": int(var.varValue)
                })
        
        # Calculate shortage reduction
        total_shortage_before = 0
        total_shortage_after = 0
        
        for shelter in shelters:
            shelter_id = shelter['id']
            if shelter_id not in forecasts:
                continue
                
            for item in items:
                predicted_need = forecasts[shelter_id][item]
                current_inventory = shelter[f'current_{item}']
                
                # Before optimization
                shortage_before = max(0, predicted_need - current_inventory)
                total_shortage_before += shortage_before
                
                # After optimization
                shortage_var_name = f"shortage_{shelter_id}_{item}"
                if shortage_var_name in shortage_vars:
                    shortage_after = shortage_vars[shortage_var_name].varValue
                    total_shortage_after += shortage_after
        
        shortage_reduction = 0.0
        if total_shortage_before > 0:
            shortage_reduction = ((total_shortage_before - total_shortage_after) / total_shortage_before) * 100
        
        return transfers_list, shortage_reduction
    
    def simple_optimize(self, shelters: List[Dict], forecasts: Dict[str, Dict]) -> Tuple[List[Dict], float]:
        """
        Simplified optimization that only transfers from surplus to shortage
        """
        transfers = []
        total_shortage_before = 0
        total_shortage_after = 0
        
        # Calculate shortages and surpluses
        shortages = {}
        surpluses = {}
        
        for shelter in shelters:
            shelter_id = shelter['id']
            if shelter_id not in forecasts:
                continue
                
            shortages[shelter_id] = {}
            surpluses[shelter_id] = {}
            
            for item in ['beds', 'meals', 'kits']:
                predicted_need = forecasts[shelter_id][item]
                current_inventory = shelter[f'current_{item}']
                
                if predicted_need > current_inventory:
                    shortage = predicted_need - current_inventory
                    shortages[shelter_id][item] = shortage
                    total_shortage_before += shortage
                else:
                    surplus = current_inventory - predicted_need
                    surpluses[shelter_id][item] = surplus
        
        # Find transfers
        for shelter_from in shelters:
            shelter_from_id = shelter_from['id']
            if shelter_from_id not in surpluses:
                continue
                
            for shelter_to in shelters:
                shelter_to_id = shelter_to['id']
                if shelter_to_id == shelter_from_id or shelter_to_id not in shortages:
                    continue
                
                for item in ['beds', 'meals', 'kits']:
                    if (item in surpluses[shelter_from_id] and 
                        item in shortages[shelter_to_id] and
                        surpluses[shelter_from_id][item] > 0 and
                        shortages[shelter_to_id][item] > 0):
                        
                        transfer_amount = min(
                            surpluses[shelter_from_id][item],
                            shortages[shelter_to_id][item]
                        )
                        
                        if transfer_amount > 0:
                            transfers.append({
                                "from": shelter_from_id,
                                "to": shelter_to_id,
                                "item": item,
                                "amount": int(transfer_amount)
                            })
                            
                            # Update surpluses and shortages
                            surpluses[shelter_from_id][item] -= transfer_amount
                            shortages[shelter_to_id][item] -= transfer_amount
                            
                            total_shortage_after += max(0, shortages[shelter_to_id][item])
        
        shortage_reduction = 0.0
        if total_shortage_before > 0:
            shortage_reduction = ((total_shortage_before - total_shortage_after) / total_shortage_before) * 100
        
        return transfers, shortage_reduction 