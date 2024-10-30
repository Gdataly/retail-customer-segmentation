import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RetailDataGenerator:
    """
    Generates realistic retail transaction data based on common retail patterns
    """
    
    def __init__(self, start_date='2023-01-01', end_date='2023-12-31', num_customers=1000):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.num_customers = num_customers
        
        # Validate date range
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        
        self.date_range_days = (self.end_date - self.start_date).days
        
        # Define customer segments and their characteristics
        self.customer_segments = {
            'loyal_high_value': 0.15,    # 15% loyal high-value customers
            'regular_medium': 0.30,       # 30% regular medium-value customers
            'occasional_low': 0.35,       # 35% occasional buyers
            'new_customers': 0.15,        # 15% new customers
            'inactive': 0.05              # 5% inactive customers
        }
        
        # Define product categories and price ranges
        self.product_categories = {
            'electronics': {'min_price': 100, 'max_price': 2000, 'frequency': 0.15},
            'clothing': {'min_price': 20, 'max_price': 200, 'frequency': 0.30},
            'groceries': {'min_price': 5, 'max_price': 100, 'frequency': 0.35},
            'home_garden': {'min_price': 30, 'max_price': 500, 'frequency': 0.10},
            'beauty': {'min_price': 10, 'max_price': 150, 'frequency': 0.10}
        }

    def _generate_customer_base(self):
        """Generate customer base with segments"""
        customers = []
        customer_ids = range(1, self.num_customers + 1)
        
        for customer_id in customer_ids:
            segment = np.random.choice(
                list(self.customer_segments.keys()),
                p=list(self.customer_segments.values())
            )
            
            # Assign joining date based on segment
            if segment == 'new_customers':
                # New customers join in the last 90 days before end_date
                days_before_end = min(90, self.date_range_days)
                join_date = self.end_date - timedelta(days=np.random.randint(0, days_before_end))
            elif segment == 'inactive':
                # Inactive customers joined near the start date
                days_after_start = min(90, self.date_range_days)
                join_date = self.start_date + timedelta(days=np.random.randint(0, days_after_start))
            else:
                # Other customers joined before the start date
                join_date = self.start_date - timedelta(days=np.random.randint(0, 365))
                
            customers.append({
                'customer_id': customer_id,
                'segment': segment,
                'join_date': join_date
            })
            
        return pd.DataFrame(customers)

    def _generate_purchase_patterns(self, customer_data):
        """Generate purchase patterns based on customer segments"""
        transactions = []
        invoice_no = 1
        
        for _, customer in customer_data.iterrows():
            # Determine number of transactions based on segment
            if customer['segment'] == 'loyal_high_value':
                num_transactions = np.random.randint(24, 48)  # 2-4 purchases per month
            elif customer['segment'] == 'regular_medium':
                num_transactions = np.random.randint(12, 24)  # 1-2 purchases per month
            elif customer['segment'] == 'occasional_low':
                num_transactions = np.random.randint(4, 12)   # 4-12 purchases per year
            elif customer['segment'] == 'new_customers':
                num_transactions = np.random.randint(1, 4)    # 1-4 purchases since joining
            else:  # inactive
                num_transactions = np.random.randint(0, 2)    # 0-1 purchases
                
            # Generate transactions for this customer
            for _ in range(num_transactions):
                # Determine purchase date
                if customer['segment'] == 'new_customers':
                    # Ensure at least 1 day difference
                    date_diff = max(1, (self.end_date - customer['join_date']).days)
                    purchase_date = customer['join_date'] + timedelta(
                        days=np.random.randint(0, date_diff)
                    )
                elif customer['segment'] == 'inactive':
                    # Inactive customers purchase within 90 days of joining
                    date_diff = min(90, (self.end_date - customer['join_date']).days)
                    purchase_date = customer['join_date'] + timedelta(
                        days=np.random.randint(0, date_diff)
                    )
                else:
                    # Regular customers purchase throughout the date range
                    purchase_date = self.start_date + timedelta(
                        days=np.random.randint(0, self.date_range_days + 1)
                    )
                
                # Generate items in this transaction
                num_items = np.random.randint(1, 8)
                for _ in range(num_items):
                    category = np.random.choice(
                        list(self.product_categories.keys()),
                        p=[cat['frequency'] for cat in self.product_categories.values()]
                    )
                    
                    base_price = np.random.uniform(
                        self.product_categories[category]['min_price'],
                        self.product_categories[category]['max_price']
                    )
                    
                    # Apply discounts based on segment
                    if customer['segment'] == 'loyal_high_value':
                        discount = np.random.uniform(0.1, 0.2)  # 10-20% discount
                    elif customer['segment'] == 'new_customers':
                        discount = np.random.uniform(0.15, 0.25)  # 15-25% discount
                    else:
                        discount = np.random.uniform(0, 0.15)  # 0-15% discount
                        
                    final_price = base_price * (1 - discount)
                    
                    transactions.append({
                        'invoice_no': invoice_no,
                        'customer_id': customer['customer_id'],
                        'date': purchase_date,
                        'category': category,
                        'amount': round(final_price, 2),
                        'discount_applied': round(discount * 100, 2)
                    })
                
                invoice_no += 1
                
        return pd.DataFrame(transactions)

    def generate_data(self):
        """Generate complete retail dataset"""
        # Generate customer base
        customer_data = self._generate_customer_base()
        
        # Generate transactions
        transactions = self._generate_purchase_patterns(customer_data)
        
        # Sort transactions by date
        transactions = transactions.sort_values('date').reset_index(drop=True)
        
        return customer_data, transactions

    def save_data(self, output_dir='data/raw/'):
        """Save generated data to CSV files"""
        customer_data, transactions = self.generate_data()
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save files
        customer_data.to_csv(f'{output_dir}customers.csv', index=False)
        transactions.to_csv(f'{output_dir}transactions.csv', index=False)
        
        return {
            'customers_file': f'{output_dir}customers.csv',
            'transactions_file': f'{output_dir}transactions.csv',
            'customers_shape': customer_data.shape,
            'transactions_shape': transactions.shape
        }

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = RetailDataGenerator(
        start_date='2023-01-01',
        end_date='2023-12-31',
        num_customers=1000
    )
    
    # Generate and save data
    result = generator.save_data()
    
    print("Data Generation Summary:")
    print(f"Number of customers: {result['customers_shape'][0]}")
    print(f"Number of transactions: {result['transactions_shape'][0]}")
    print("\nFiles saved:")
    print(f"Customers data: {result['customers_file']}")
    print(f"Transactions data: {result['transactions_file']}")
