import sys
import pandas as pd

def validate_combined_schema():
    print("--- Step 2: Running Automated Data Validation Gatekeeper ---")
    data_path = 'data/AzureLLMInferenceTrace_combined.csv'
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"❌ Structural Failure: Could not read merged file. Reason: {e}")
        sys.exit(1) # Exit code 1 completely freezes and fails the GitHub Action runner

    # 1. Column Integrity Validation
    expected_schema = ['TIMESTAMP', 'ContextTokens', 'GeneratedTokens', 'WorkloadType']
    for col in expected_schema:
        if col not in df.columns:
            print(f"❌ Schema Failure: Required feature column '{col}' is missing.")
            sys.exit(1)
            
    # 2. Anomaly Checking: Negative dimensions
    if (df['ContextTokens'] < 0).any() or (df['GeneratedTokens'] < 0).any():
        print("❌ Data Anomaly Failure: Found negative token values in execution history traces.")
        sys.exit(1)
        
    # 3. Size Adequacy Checking
    if len(df) < 500:
        print("❌ Training Hazard: Merged trace count is too sparse to calibrate models safely.")
        sys.exit(1)

    print("✅ Validation Success: Combined trace schema meets all production criteria.")
    sys.exit(0) # Exit code 0 signals GitHub that it is safe to proceed to training

if __name__ == "__main__":
    validate_combined_schema()