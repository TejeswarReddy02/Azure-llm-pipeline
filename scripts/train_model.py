import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_production_model():
    print("--- Step 3: Starting Production Model Training ---")
    data_path = 'data/AzureLLMInferenceTrace_combined.csv'
    
    # 1. Load the verified combined dataset
    df = pd.read_csv(data_path, parse_dates=['TIMESTAMP'])
    
    # 2. Feature Engineering: Extract the hour of day from the timestamp
    df['Hour'] = df['TIMESTAMP'].dt.hour
    
    # Define our inputs (Features) and output (Target)
    X = df[['ContextTokens', 'WorkloadType', 'Hour']]
    y = df['GeneratedTokens']
    
    # 3. Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and Train a fast, reliable Random Forest Regressor
    print(f"Training model on {len(X_train)} production request traces...")
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 5. Evaluate accuracy (Mean Absolute Error)
    train_score = model.score(X_train, y_train)
    print(f"✅ Training complete! Model R^2 Score: {train_score:.4f}")
    
    # 6. Save the trained model binary (Production Ready Artifact)
    os.makedirs('models', exist_ok=True)
    model_output_path = 'models/llm_workload_predictor.pkl'
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"📦 Production artifact saved successfully to: {model_output_path}")

if __name__ == "__main__":
    train_production_model()