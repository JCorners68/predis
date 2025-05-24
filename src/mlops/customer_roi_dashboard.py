#!/usr/bin/env python3
"""
Customer ROI Dashboard - Real-time value tracking for Predis deployments
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import statistics

class CustomerROICalculator:
    """Calculate real-time ROI metrics for Predis customers"""
    
    def __init__(self, customer_id: str, baseline_metrics: Dict):
        self.customer_id = customer_id
        self.baseline_metrics = baseline_metrics
        self.current_metrics = {}
        self.roi_history = []
        
    def calculate_gpu_utilization_improvement(self) -> float:
        """Calculate GPU utilization improvement from reduced data loading time"""
        baseline_gpu = self.baseline_metrics.get('gpu_utilization', 0.6)
        current_gpu = self.current_metrics.get('gpu_utilization', 0.85)
        
        improvement = ((current_gpu - baseline_gpu) / baseline_gpu) * 100
        return improvement
    
    def calculate_training_time_reduction(self) -> float:
        """Calculate ML training time reduction"""
        baseline_time = self.baseline_metrics.get('avg_training_time_hours', 10.0)
        current_time = self.current_metrics.get('avg_training_time_hours', 6.0)
        
        reduction = ((baseline_time - current_time) / baseline_time) * 100
        return reduction
    
    def calculate_infrastructure_cost_savings(self) -> Dict[str, float]:
        """Calculate infrastructure cost savings"""
        # GPU instance costs (per hour)
        gpu_costs = {
            'p3.2xlarge': 3.06,
            'p3.8xlarge': 12.24,
            'g4dn.xlarge': 0.526,
            'g4dn.2xlarge': 0.752
        }
        
        instance_type = self.baseline_metrics.get('instance_type', 'p3.2xlarge')
        hourly_cost = gpu_costs.get(instance_type, 3.06)
        
        # Calculate savings from reduced training time
        time_saved_pct = self.calculate_training_time_reduction()
        daily_training_hours = self.baseline_metrics.get('daily_training_hours', 20)
        
        daily_savings = (daily_training_hours * hourly_cost * time_saved_pct / 100)
        monthly_savings = daily_savings * 30
        annual_savings = daily_savings * 365
        
        return {
            'daily_savings_usd': round(daily_savings, 2),
            'monthly_savings_usd': round(monthly_savings, 2),
            'annual_savings_usd': round(annual_savings, 2),
            'roi_days': round(self.baseline_metrics.get('predis_cost', 10000) / daily_savings, 1)
        }
    
    def calculate_revenue_impact(self) -> Dict[str, float]:
        """Calculate revenue impact from faster processing"""
        base_revenue = self.baseline_metrics.get('monthly_revenue', 1000000)
        
        # Different impacts by industry
        industry_multipliers = {
            'ml_training': 0.15,  # 15% revenue increase from faster model deployment
            'hft': 0.25,         # 25% from better trade execution
            'gaming': 0.10,      # 10% from improved user experience
            'streaming': 0.08    # 8% from reduced buffering
        }
        
        industry = self.baseline_metrics.get('industry', 'ml_training')
        multiplier = industry_multipliers.get(industry, 0.10)
        
        # Calculate based on performance improvement
        perf_improvement = self.current_metrics.get('avg_speedup', 15.0)
        revenue_impact_pct = min(perf_improvement * multiplier / 10, 50)  # Cap at 50%
        
        monthly_impact = base_revenue * revenue_impact_pct / 100
        annual_impact = monthly_impact * 12
        
        return {
            'revenue_increase_pct': round(revenue_impact_pct, 1),
            'monthly_revenue_impact': round(monthly_impact, 0),
            'annual_revenue_impact': round(annual_impact, 0)
        }
    
    def generate_roi_summary(self) -> Dict:
        """Generate comprehensive ROI summary"""
        gpu_improvement = self.calculate_gpu_utilization_improvement()
        training_reduction = self.calculate_training_time_reduction()
        cost_savings = self.calculate_infrastructure_cost_savings()
        revenue_impact = self.calculate_revenue_impact()
        
        total_annual_value = (cost_savings['annual_savings_usd'] + 
                            revenue_impact['annual_revenue_impact'])
        
        return {
            'customer_id': self.customer_id,
            'timestamp': datetime.utcnow().isoformat(),
            'performance_metrics': {
                'gpu_utilization_improvement_pct': round(gpu_improvement, 1),
                'training_time_reduction_pct': round(training_reduction, 1),
                'avg_speedup_factor': self.current_metrics.get('avg_speedup', 15.0),
                'cache_hit_rate_improvement_pct': self.current_metrics.get('hit_rate_improvement', 22.3)
            },
            'cost_savings': cost_savings,
            'revenue_impact': revenue_impact,
            'total_value': {
                'annual_value_usd': round(total_annual_value, 0),
                'roi_percentage': round((total_annual_value / self.baseline_metrics.get('predis_cost', 10000)) * 100, 0),
                'payback_period_days': cost_savings['roi_days']
            }
        }
    
    def update_metrics(self, new_metrics: Dict):
        """Update current metrics"""
        self.current_metrics.update(new_metrics)
        roi_summary = self.generate_roi_summary()
        self.roi_history.append(roi_summary)
        
        # Keep last 30 days of history
        cutoff = datetime.utcnow() - timedelta(days=30)
        self.roi_history = [r for r in self.roi_history 
                          if datetime.fromisoformat(r['timestamp']) > cutoff]


class ROIDashboardServer:
    """Web server for ROI dashboard"""
    
    def __init__(self, port: int = 8889):
        self.port = port
        self.calculators: Dict[str, CustomerROICalculator] = {}
        
    def add_customer(self, customer_id: str, baseline_metrics: Dict):
        """Add new customer to tracking"""
        self.calculators[customer_id] = CustomerROICalculator(customer_id, baseline_metrics)
    
    def generate_html_dashboard(self, customer_id: str) -> str:
        """Generate HTML dashboard for customer"""
        if customer_id not in self.calculators:
            return "<h1>Customer not found</h1>"
            
        calc = self.calculators[customer_id]
        roi = calc.generate_roi_summary()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Predis ROI Dashboard - {customer_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: auto; }}
                .card {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 36px; font-weight: bold; color: #2196F3; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .savings {{ color: #4CAF50; }}
                .chart {{ width: 100%; height: 300px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; margin-top: 30px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f0f0f0; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="container">
                <h1>Predis ROI Dashboard</h1>
                <p>Customer: {customer_id} | Updated: {roi['timestamp']}</p>
                
                <div class="card">
                    <h2>Key Performance Metrics</h2>
                    <div class="metric">
                        <div class="metric-value">{roi['performance_metrics']['avg_speedup_factor']}x</div>
                        <div class="metric-label">Average Speedup</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{roi['performance_metrics']['gpu_utilization_improvement_pct']}%</div>
                        <div class="metric-label">GPU Utilization Improvement</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{roi['performance_metrics']['training_time_reduction_pct']}%</div>
                        <div class="metric-label">Training Time Reduction</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{roi['performance_metrics']['cache_hit_rate_improvement_pct']}%</div>
                        <div class="metric-label">Cache Hit Rate Improvement</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Financial Impact</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Daily</th>
                            <th>Monthly</th>
                            <th>Annual</th>
                        </tr>
                        <tr>
                            <td>Infrastructure Cost Savings</td>
                            <td class="savings">${roi['cost_savings']['daily_savings_usd']:,.2f}</td>
                            <td class="savings">${roi['cost_savings']['monthly_savings_usd']:,.2f}</td>
                            <td class="savings">${roi['cost_savings']['annual_savings_usd']:,.2f}</td>
                        </tr>
                        <tr>
                            <td>Revenue Impact ({roi['revenue_impact']['revenue_increase_pct']}% increase)</td>
                            <td class="savings">${roi['revenue_impact']['monthly_revenue_impact']/30:,.2f}</td>
                            <td class="savings">${roi['revenue_impact']['monthly_revenue_impact']:,.2f}</td>
                            <td class="savings">${roi['revenue_impact']['annual_revenue_impact']:,.2f}</td>
                        </tr>
                        <tr style="font-weight: bold;">
                            <td>Total Value</td>
                            <td class="savings">${roi['total_value']['annual_value_usd']/365:,.2f}</td>
                            <td class="savings">${roi['total_value']['annual_value_usd']/12:,.2f}</td>
                            <td class="savings">${roi['total_value']['annual_value_usd']:,.2f}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="card">
                    <h2>ROI Summary</h2>
                    <div class="metric">
                        <div class="metric-value savings">{roi['total_value']['roi_percentage']}%</div>
                        <div class="metric-label">Annual ROI</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{roi['total_value']['payback_period_days']} days</div>
                        <div class="metric-label">Payback Period</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Performance Trend (Last 7 Days)</h2>
                    <canvas id="trendChart" class="chart"></canvas>
                </div>
            </div>
            
            <script>
                // Mock trend data - in production, fetch from API
                const ctx = document.getElementById('trendChart').getContext('2d');
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
                        datasets: [{{
                            label: 'Speedup Factor',
                            data: [12.5, 13.2, 14.1, 14.8, 15.2, 15.0, 15.3],
                            borderColor: '#2196F3',
                            tension: 0.1
                        }}, {{
                            label: 'GPU Utilization %',
                            data: [78, 80, 82, 84, 85, 85, 86],
                            borderColor: '#4CAF50',
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false
                    }}
                }});
                
                // Auto-refresh every 30 seconds
                setTimeout(() => location.reload(), 30000);
            </script>
        </body>
        </html>
        """
        
        return html


# Example usage
if __name__ == "__main__":
    # Example customer baseline metrics
    ml_training_customer = {
        'instance_type': 'p3.8xlarge',
        'gpu_utilization': 0.60,
        'avg_training_time_hours': 10.0,
        'daily_training_hours': 20,
        'monthly_revenue': 2000000,
        'industry': 'ml_training',
        'predis_cost': 50000
    }
    
    # Create calculator
    calc = CustomerROICalculator('customer-001', ml_training_customer)
    
    # Update with current metrics
    calc.update_metrics({
        'gpu_utilization': 0.85,
        'avg_training_time_hours': 6.0,
        'avg_speedup': 15.0,
        'hit_rate_improvement': 22.3
    })
    
    # Generate and print ROI summary
    roi = calc.generate_roi_summary()
    print(json.dumps(roi, indent=2))