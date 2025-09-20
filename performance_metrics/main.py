import psycopg2
import pandas as pd
from io import StringIO
import csv
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import re

sys.path.append(str(Path(__file__).parent))
from analysis import LedgerAnalyzer

def create_ledger_status_table():
    """Create schema and table without unwanted columns"""
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="exchange",
            user="postgres",
            password="shah7788",
            port=5432
        )
        cursor = connection.cursor()
        
        # Create schema if not exists
        cursor.execute("CREATE SCHEMA IF NOT EXISTS ledger_status")
        
        # Create base table with only strategy_name
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ledger_status.strategy_metrics (
            strategy_name VARCHAR(50) PRIMARY KEY
        )
        """)
        
        connection.commit()
        print("✅ Created ledger_status.strategy_metrics base table")
        
    except Exception as e:
        print(f"Error creating status table: {e}")
        raise
    finally:
        if connection:
            connection.close()

def create_dynamic_columns(cursor, metrics_dict):
    """Dynamically add columns to the table based on the metrics dictionary"""
    type_mapping = {
        'int': 'INTEGER',
        'float': 'DECIMAL(15,4)',
        'bool': 'BOOLEAN',
        'str': 'TEXT'
    }
    
    for metric_name, value in metrics_dict.items():
        col_name = re.sub(r'[^a-zA-Z0-9_]', '_', metric_name).lower()
        
        if isinstance(value, (int, np.integer)):
            col_type = type_mapping['int']
        elif isinstance(value, (float, np.floating)):
            col_type = type_mapping['float']
        elif isinstance(value, bool):
            col_type = type_mapping['bool']
        else:
            col_type = type_mapping['str']
        
        try:
            cursor.execute(f"""
            ALTER TABLE ledger_status.strategy_metrics 
            ADD COLUMN IF NOT EXISTS {col_name} {col_type}
            """)
        except Exception as e:
            print(f"Warning: Could not add column {col_name}: {e}")

def save_strategy_metrics(strategy_name, stats, error=None):
    """Save all metrics to individual columns, excluding duplicates and unnecessary fields."""
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="exchange",
            user="postgres",
            password="shah7788",
            port=5432
        )
        cursor = connection.cursor()

        # Metrics to exclude
        excluded_keys = {
            'avg_win', 'avg_loss', 'max_win', 'max_loss',
            'median_win', 'median_loss', 'win_std_dev', 'loss_std_dev'
        }

        filtered_stats = {k: v for k, v in stats.items() if k not in excluded_keys}
        create_dynamic_columns(cursor, filtered_stats)
        connection.commit()

        # Build insert columns
        columns = ['strategy_name']
        values = [strategy_name]

        for metric_name, value in filtered_stats.items():
            col_name = re.sub(r'[^a-zA-Z0-9_]', '_', metric_name).lower()

            if hasattr(value, 'item'):
                value = value.item()
            elif isinstance(value, (np.integer)):
                value = int(value)
            elif isinstance(value, (np.floating)):
                value = float(value)
            elif value is None:
                value = None
            elif isinstance(value, (list, dict)):
                value = str(value)

            columns.append(col_name)
            values.append(value)

        # Upsert
        cols_str = ', '.join(columns)
        vals_str = ', '.join(['%s'] * len(values))
        update_str = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns[1:]])

        query = f"""
        INSERT INTO ledger_status.strategy_metrics ({cols_str})
        VALUES ({vals_str})
        ON CONFLICT (strategy_name) 
        DO UPDATE SET {update_str}
        """

        print("📦 Saving columns:", columns)
        cursor.execute(query, values)
        connection.commit()
        print(f"✅ Saved metrics for {strategy_name}")

    except Exception as e:
        print(f"🔥 Failed to save metrics for {strategy_name}: {e}")
        raise
    finally:
        if connection:
            connection.close()

def fetch_ledger_data(strategy_name):
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="exchange",
            user="postgres",
            password="shah7788",
            port=5432
        )
        cursor = connection.cursor()
        
        query = f"SELECT * FROM ledger.{strategy_name} ORDER BY datetime"
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        return columns, data
        
    except Exception as e:
        print(f"Error fetching data for {strategy_name}: {e}")
        raise
    finally:
        if connection:
            connection.close()

def analyze_strategy(strategy_name):
    print(f"\n🔍 Analyzing {strategy_name}...")
    try:
        columns, data = fetch_ledger_data(strategy_name)
        print("✅ Data fetched for:", strategy_name)
        
        if not data:
            print(f"No data found for {strategy_name}")
            save_strategy_metrics(strategy_name, {}, error="No data found")
            return
        
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(columns)
        writer.writerows(data)
        csv_buffer.seek(0)
        
        analyzer = LedgerAnalyzer(csv_buffer)
        analyzer.set_initial_capital(1000)
        analyzer.preprocess()
        stats = analyzer.calculate_all_stats()
        
        print("\n📊 Trading Performance Summary:")
        for category in [
            'net_profit', 'annualized_return', 'sharpe_ratio', 
            'max_drawdown', 'win_rate', 'profit_factor'
        ]:
            print(f"{category.replace('_', ' ').title()}: {stats.get(category, 'N/A')}")

        print("\n📊 Full Trading Performance Metrics:")
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        save_strategy_metrics(strategy_name, stats)
        
    except Exception as e:
        print(f"🔥 Analysis failed for {strategy_name}: {e}")
        save_strategy_metrics(strategy_name, {}, error=str(e))
        raise

def get_strategy_metrics(strategy_name):
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="exchange",
            user="postgres",
            password="shah7788",
            port=5432
        )
        cursor = connection.cursor()
        
        cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'ledger_status' 
        AND table_name = 'strategy_metrics'
        """)
        columns = [row[0] for row in cursor.fetchall()]
        
        cols_str = ', '.join(columns)
        cursor.execute(f"""
        SELECT {cols_str}
        FROM ledger_status.strategy_metrics
        WHERE strategy_name = %s
        """, (strategy_name,))
        
        result = cursor.fetchone()
        if result:
            return dict(zip(columns, result))
        return None
        
    except Exception as e:
        print(f"Error retrieving metrics: {e}")
        return None
    finally:
        if connection:
            connection.close()

def main():
    create_ledger_status_table()
    
    strategies = [f"backtest_strategy_{i:02d}" for i in range(108)]
    
    for strategy in strategies:
        try:
            analyze_strategy(strategy)
        except Exception as e:
            print(f"🔥 Critical error processing {strategy}: {e}")
            continue
    
    print("\n✅ Analysis complete. Summary of saved results:")
    for strategy in strategies:
        metrics = get_strategy_metrics(strategy)
        if metrics:
            net_profit = metrics.get('net_profit', 'N/A')
            print(f"{strategy}: Net Profit: {net_profit}")

if __name__ == "__main__":
    main()



