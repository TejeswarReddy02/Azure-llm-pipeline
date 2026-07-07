import os
import pandas as pd

def merge_datasets():
    print("--- Step 1: Starting Data Merge Phase ---")
    
    code_path = 'data/AzureLLMInferenceTrace_code.csv'
    conv_path = 'data/AzureLLMInferenceTrace_conv.csv'
    output_path = 'data/AzureLLMInferenceTrace_combined.csv'
    
    # Validation check to ensure you placed files in the right folder
    if not os.path.exists(code_path) or not os.path.exists(conv_path):
        print("❌ Error: Missing raw source files in your data/ directory.")
        return False
        
    # Load separate trace logs
    code_df = pd.read_csv(code_path)
    conv_df = pd.read_csv(conv_path)
    
    # Inject the WorkloadType context feature (1 = Code trace, 0 = Chat conversation)
    code_df['WorkloadType'] = 1
    conv_df['WorkloadType'] = 0
    
    # Merge them vertically
    combined_df = pd.concat([code_df, conv_df], ignore_index=True)
    
    # Shuffle completely so the training data is mixed
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the consolidated tracking trace back to data directory
    os.makedirs('data', exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"✅ Success: Generated combined file with {len(combined_df)} entries.")
    return True

if __name__ == "__main__":
    merge_datasets()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
