import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_logons(n_records=10000, anomaly_rate=0.05):
    """
    Generate synthetic VDI logon data.
    
    Parameters:
    - n_records: total logon events to generate
    - anomaly_rate: percentage of anomalous logons (0-1)
    
    Returns:
    - DataFrame with logon metrics
    """
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    data = {
        'timestamp': [],
        'user_id': [],
        'machine_id': [],
        'logon_duration_sec': [],
        'profile_load_time_sec': [],
        'auth_time_sec': [],
        'desktop_init_time_sec': [],
        'concurrent_logons': [],
        'machine_cpu_during_logon': [],
        'machine_memory_during_logon': [],
        'is_anomaly': []
    }
    
    # Generate base timestamp
    start_time = datetime(2026, 1, 1, 8, 0, 0)  # Business hours start
    
    n_normal = int(n_records * (1 - anomaly_rate))
    n_anomaly = n_records - n_normal
    
    # NORMAL logons
    for i in range(n_normal):
        data['timestamp'].append(start_time + timedelta(seconds=np.random.randint(0, 86400*30)))
        data['user_id'].append(f"USER_{np.random.randint(1, 500)}")
        data['machine_id'].append(f"DESKTOP_{np.random.randint(1, 200)}")
        
        # Normal logon times (based on Citrix handbook)
        data['logon_duration_sec'].append(np.random.normal(45, 10))  # Mean 45s, std 10s
        data['profile_load_time_sec'].append(np.random.normal(15, 3))
        data['auth_time_sec'].append(np.random.normal(5, 1))
        data['desktop_init_time_sec'].append(np.random.normal(25, 5))
        
        # Normal concurrent logons (business hours variation)
        data['concurrent_logons'].append(np.random.poisson(15))  # Avg 15 concurrent
        
        # Normal machine load
        data['machine_cpu_during_logon'].append(np.random.normal(50, 15))
        data['machine_memory_during_logon'].append(np.random.normal(65, 10))
        
        data['is_anomaly'].append(0)
    
    # ANOMALOUS logons
    for i in range(n_anomaly):
        data['timestamp'].append(start_time + timedelta(seconds=np.random.randint(0, 86400*30)))
        data['user_id'].append(f"USER_{np.random.randint(1, 500)}")
        data['machine_id'].append(f"DESKTOP_{np.random.randint(1, 200)}")
        
        # Pick an anomaly type
        anomaly_type = np.random.choice(['slow_logon', 'boot_storm', 'high_load', 'network_issue'])
        
        if anomaly_type == 'slow_logon':
            # Logon takes much longer
            data['logon_duration_sec'].append(np.random.normal(180, 30))  # Mean 180s (3x normal)
            data['profile_load_time_sec'].append(np.random.normal(80, 20))
            data['auth_time_sec'].append(np.random.normal(8, 2))
            data['desktop_init_time_sec'].append(np.random.normal(92, 20))
            data['concurrent_logons'].append(np.random.poisson(12))
            data['machine_cpu_during_logon'].append(np.random.normal(70, 15))
            data['machine_memory_during_logon'].append(np.random.normal(80, 10))
            
        elif anomaly_type == 'boot_storm':
            # Many concurrent logons
            data['logon_duration_sec'].append(np.random.normal(95, 20))
            data['profile_load_time_sec'].append(np.random.normal(40, 10))
            data['auth_time_sec'].append(np.random.normal(6, 1))
            data['desktop_init_time_sec'].append(np.random.normal(49, 15))
            data['concurrent_logons'].append(np.random.poisson(60))  # 4x normal
            data['machine_cpu_during_logon'].append(np.random.normal(85, 10))
            data['machine_memory_during_logon'].append(np.random.normal(88, 8))
            
        elif anomaly_type == 'high_load':
            # Machine under heavy load
            data['logon_duration_sec'].append(np.random.normal(120, 25))
            data['profile_load_time_sec'].append(np.random.normal(50, 15))
            data['auth_time_sec'].append(np.random.normal(7, 2))
            data['desktop_init_time_sec'].append(np.random.normal(63, 15))
            data['concurrent_logons'].append(np.random.poisson(20))
            data['machine_cpu_during_logon'].append(np.random.normal(92, 5))  # Very high CPU
            data['machine_memory_during_logon'].append(np.random.normal(95, 3))  # Very high memory
            
        else:  # network_issue
            # Auth and profile load slow (network bottleneck)
            data['logon_duration_sec'].append(np.random.normal(140, 30))
            data['profile_load_time_sec'].append(np.random.normal(70, 20))  # Profile slow
            data['auth_time_sec'].append(np.random.normal(15, 5))  # Auth slow (network)
            data['desktop_init_time_sec'].append(np.random.normal(30, 8))
            data['concurrent_logons'].append(np.random.poisson(16))
            data['machine_cpu_during_logon'].append(np.random.normal(55, 15))
            data['machine_memory_during_logon'].append(np.random.normal(68, 12))
        
        data['is_anomaly'].append(1)
    
    df = pd.DataFrame(data)
    
    # Ensure no negative values
    numeric_cols = ['logon_duration_sec', 'profile_load_time_sec', 'auth_time_sec', 
                    'desktop_init_time_sec', 'concurrent_logons', 
                    'machine_cpu_during_logon', 'machine_memory_during_logon']
    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    df = generate_synthetic_logons(n_records=10000, anomaly_rate=0.05)
    df.to_csv('data/synthetic_logons.csv', index=False)
    print(f"Generated {len(df)} logon records")
    print(f"\nFirst few records:\n{df.head()}")
    print(f"\nDataset info:\n{df.info()}")
    print(f"\nAnomaly distribution:\n{df['is_anomaly'].value_counts()}")