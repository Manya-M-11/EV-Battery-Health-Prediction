# EV Battery Life Prediction System
# Using NASA Battery Dataset with ML & GenAI

"""
Step 1: Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn xgboost scipy
pip install openai langchain pinecone-client
pip install flask flask-cors
pip install scipy h5py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

class NASABatteryDataProcessor:
    """Process NASA Battery Dataset for ML training"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.processed_data = None
        
    def load_matlab_file(self, filename):
        """Load MATLAB .mat file from NASA dataset"""
        try:
            mat_data = loadmat(filename)
            return mat_data
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def extract_features_from_cycle(self, cycle_data):
        """Extract features from a single charge/discharge cycle"""
        features = {}
        
        try:
            # Extract voltage, current, temperature data
            if 'data' in cycle_data.dtype.names:
                data = cycle_data['data'][0, 0]
                
                # Voltage features
                if 'Voltage_measured' in data.dtype.names:
                    voltage = data['Voltage_measured'][0, 0].flatten()
                    features['voltage_mean'] = np.mean(voltage)
                    features['voltage_std'] = np.std(voltage)
                    features['voltage_min'] = np.min(voltage)
                    features['voltage_max'] = np.max(voltage)
                    features['voltage_range'] = np.max(voltage) - np.min(voltage)
                
                # Current features
                if 'Current_measured' in data.dtype.names:
                    current = data['Current_measured'][0, 0].flatten()
                    features['current_mean'] = np.mean(current)
                    features['current_std'] = np.std(current)
                
                # Temperature features
                if 'Temperature_measured' in data.dtype.names:
                    temp = data['Temperature_measured'][0, 0].flatten()
                    features['temp_mean'] = np.mean(temp)
                    features['temp_std'] = np.std(temp)
                    features['temp_max'] = np.max(temp)
                
                # Capacity (if available)
                if 'Capacity' in data.dtype.names:
                    capacity = data['Capacity'][0, 0].flatten()
                    if len(capacity) > 0:
                        features['capacity'] = capacity[0]
            
            # Ambient temperature
            if 'ambient_temperature' in cycle_data.dtype.names:
                features['ambient_temp'] = cycle_data['ambient_temperature'][0, 0][0, 0]
                
        except Exception as e:
            print(f"Error extracting features: {e}")
        
        return features
    
    def process_battery_file(self, filename, battery_id):
        """Process a single battery file and extract time-series features"""
        mat_data = self.load_matlab_file(filename)
        if mat_data is None:
            return None
        
        # Get the main data structure (usually the battery name)
        key = [k for k in mat_data.keys() if not k.startswith('__')][0]
        battery_data = mat_data[key]
        
        cycles_features = []
        
        # Process each cycle
        for cycle_idx in range(battery_data.shape[1]):
            cycle = battery_data[0, cycle_idx]
            features = self.extract_features_from_cycle(cycle)
            
            if features:
                features['battery_id'] = battery_id
                features['cycle_number'] = cycle_idx + 1
                
                # Calculate SOH (State of Health) based on capacity fade
                # NASA criteria: EOL at 30% fade (from 2Ah to 1.4Ah)
                if 'capacity' in features:
                    rated_capacity = 2.0  # Ah
                    features['soh'] = (features['capacity'] / rated_capacity) * 100
                    features['capacity_fade'] = rated_capacity - features['capacity']
                
                cycles_features.append(features)
        
        return pd.DataFrame(cycles_features)
    
    def create_lag_features(self, df, lag_periods=[1, 2, 3, 5]):
        """Create lagged features for time-series prediction"""
        for col in ['voltage_mean', 'current_mean', 'temp_mean', 'capacity']:
            if col in df.columns:
                for lag in lag_periods:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling statistics
        for col in ['voltage_mean', 'temp_mean', 'capacity']:
            if col in df.columns:
                df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5).mean()
                df[f'{col}_rolling_std_5'] = df[col].rolling(window=5).std()
        
        return df.dropna()


# ============================================================================
# PART 2: MACHINE LEARNING MODELS
# ============================================================================

class BatteryLifePredictor:
    """ML model for battery life prediction"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_columns = None
        
    def prepare_data(self, df, target_col='soh'):
        """Prepare features and target for training"""
        # Select relevant features
        feature_cols = [col for col in df.columns 
                       if col not in ['battery_id', 'soh', 'capacity', 'capacity_fade', 'cycle_number']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        self.feature_columns = feature_cols
        
        return X, y
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and select the best one"""
        results = {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"{name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        # Select best model based on R² score
        best_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.best_model = results[best_name]['model']
        print(f"\nBest model: {best_name}")
        
        return results
    
    def predict_rul(self, X):
        """Predict Remaining Useful Life (RUL)"""
        X_scaled = self.scaler.transform(X)
        soh_prediction = self.best_model.predict(X_scaled)
        
        # RUL estimation based on SOH
        # Assuming linear degradation: RUL proportional to (current_SOH - EOL_threshold)
        eol_threshold = 70  # 70% SOH is typically considered EOL
        rul_cycles = []
        
        for soh in soh_prediction:
            if soh <= eol_threshold:
                rul = 0
            else:
                # Estimate cycles remaining based on degradation rate
                rul = int((soh - eol_threshold) * 10)  # Rough estimate
            rul_cycles.append(rul)
        
        return soh_prediction, np.array(rul_cycles)
    
    def save_model(self, filename='battery_predictor.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='battery_predictor.pkl'):
        """Load trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {filename}")


# ============================================================================
# PART 3: GENAI INTEGRATION FOR BATTERY CARE RECOMMENDATIONS
# ============================================================================

class BatteryCareAdvisor:
    """GenAI-powered battery care recommendations"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        # Note: Replace with your OpenAI API key or use local LLM
        
    def generate_care_recommendations(self, soh, temperature, voltage, usage_pattern):
        """Generate personalized battery care recommendations"""
        
        # Rule-based recommendations (can be enhanced with LLM)
        recommendations = []
        
        # SOH-based recommendations
        if soh < 70:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Battery Health',
                'recommendation': 'Your battery is nearing end-of-life. Consider replacement planning.',
                'impact': 'Critical - Affects vehicle reliability'
            })
        elif soh < 80:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Battery Health',
                'recommendation': 'Battery showing moderate degradation. Follow optimal charging practices.',
                'impact': 'Moderate - May reduce range'
            })
        
        # Temperature-based recommendations
        if temperature > 35:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Temperature Management',
                'recommendation': 'High operating temperature detected. Park in shade, avoid fast charging in heat.',
                'impact': 'High - Accelerates degradation'
            })
        elif temperature < 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Temperature Management',
                'recommendation': 'Cold temperature affects performance. Precondition battery before driving.',
                'impact': 'Moderate - Reduces efficiency'
            })
        
        # Charging recommendations
        recommendations.append({
            'priority': 'LOW',
            'category': 'Charging Habits',
            'recommendation': 'Maintain charge between 20-80% for daily use. Full charges only for long trips.',
            'impact': 'Positive - Extends battery life'
        })
        
        recommendations.append({
            'priority': 'LOW',
            'category': 'Charging Habits',
            'recommendation': 'Avoid leaving battery at 100% or 0% for extended periods.',
            'impact': 'Positive - Reduces stress'
        })
        
        # General care tips
        recommendations.append({
            'priority': 'LOW',
            'category': 'General Care',
            'recommendation': 'Avoid frequent fast charging. Use level 2 charging when possible.',
            'impact': 'Positive - Minimizes heat stress'
        })
        
        return recommendations
    
    def generate_detailed_report(self, soh, rul, recommendations):
        """Generate a comprehensive battery health report"""
        
        report = f"""
=== BATTERY HEALTH REPORT ===

State of Health (SOH): {soh:.1f}%
Estimated Remaining Useful Life: {rul} cycles

Health Status: {'EXCELLENT' if soh > 90 else 'GOOD' if soh > 80 else 'FAIR' if soh > 70 else 'POOR'}

=== CARE RECOMMENDATIONS ===
"""
        
        for idx, rec in enumerate(recommendations, 1):
            report += f"\n{idx}. [{rec['priority']}] {rec['category']}\n"
            report += f"   → {rec['recommendation']}\n"
            report += f"   Impact: {rec['impact']}\n"
        
        return report


# ============================================================================
# PART 4: DEMO/USAGE EXAMPLE
# ============================================================================

def main():
    """Main execution flow"""
    
    print("=" * 60)
    print("EV BATTERY LIFE PREDICTION SYSTEM")
    print("=" * 60)
    
    # Step 1: Data Processing
    print("\n[1/5] Data Processing...")
    print("Note: Download NASA dataset from:")
    print("https://data.nasa.gov/dataset/Li-ion-Battery-Aging-Datasets/uj5r-zjdb")
    print("Or: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset")
    
    # For demo: creating synthetic data similar to NASA format
    print("\nGenerating synthetic demo data...")
    np.random.seed(42)
    
    cycles = 200
    demo_data = {
        'cycle_number': range(1, cycles + 1),
        'voltage_mean': 3.6 + np.random.normal(0, 0.1, cycles) - np.linspace(0, 0.3, cycles),
        'voltage_std': np.random.uniform(0.05, 0.15, cycles),
        'voltage_min': 2.5 + np.random.normal(0, 0.05, cycles),
        'voltage_max': 4.2 + np.random.normal(0, 0.05, cycles),
        'current_mean': 1.5 + np.random.normal(0, 0.2, cycles),
        'current_std': np.random.uniform(0.1, 0.3, cycles),
        'temp_mean': 25 + np.random.normal(0, 3, cycles) + np.linspace(0, 10, cycles),
        'temp_std': np.random.uniform(1, 3, cycles),
        'temp_max': 30 + np.random.normal(0, 5, cycles) + np.linspace(0, 15, cycles),
        'ambient_temp': 22 + np.random.normal(0, 2, cycles),
        'capacity': 2.0 - np.linspace(0, 0.6, cycles) + np.random.normal(0, 0.02, cycles)
    }
    
    df = pd.DataFrame(demo_data)
    df['soh'] = (df['capacity'] / 2.0) * 100
    df['voltage_range'] = df['voltage_max'] - df['voltage_min']
    
    print(f"Generated {len(df)} cycles of demo data")
    print(f"Columns: {list(df.columns)}")
    
    # Step 2: Feature Engineering
    print("\n[2/5] Feature Engineering...")
    processor = NASABatteryDataProcessor(None)
    df = processor.create_lag_features(df)
    print(f"After lag features: {len(df)} samples, {len(df.columns)} features")
    
    # Step 3: Train ML Models
    print("\n[3/5] Training ML Models...")
    predictor = BatteryLifePredictor()
    
    X, y = predictor.prepare_data(df, target_col='soh')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = predictor.train_models(X_train, y_train, X_test, y_test)
    
    # Step 4: Make Predictions
    print("\n[4/5] Making Predictions...")
    soh_pred, rul_pred = predictor.predict_rul(X_test)
    
    print(f"\nSample Predictions:")
    for i in range(min(5, len(soh_pred))):
        print(f"  Sample {i+1}: SOH = {soh_pred[i]:.1f}%, RUL = {rul_pred[i]} cycles")
    
    # Step 5: Generate Care Recommendations
    print("\n[5/5] Generating Care Recommendations...")
    advisor = BatteryCareAdvisor()
    
    # Example for one prediction
    sample_idx = 0
    sample_soh = soh_pred[sample_idx]
    sample_temp = X_test.iloc[sample_idx]['temp_mean']
    sample_voltage = X_test.iloc[sample_idx]['voltage_mean']
    
    recommendations = advisor.generate_care_recommendations(
        sample_soh, sample_temp, sample_voltage, 'normal'
    )
    
    report = advisor.generate_detailed_report(
        sample_soh, rul_pred[sample_idx], recommendations
    )
    
    print(report)
    
    # Save model
    print("\n[SAVING] Saving trained model...")
    predictor.save_model('battery_predictor.pkl')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()