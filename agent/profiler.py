import pandas as pd

def profile_dataset(df: pd.DataFrame) -> str:
    """
    Creates a summary of the dataset for the LLM.
    Includes columns, data types, missing values, and a few sample rows.
    """
    summary = []
    summary.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    summary.append("\nColumns, Data Types, and Missing Values:")
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()
        summary.append(f"- {col} (Type: {dtype}, Missing: {missing})")
        
    summary.append("\nSample Data (First 3 rows):")
    # Convert to CSV string for compact representation
    summary.append(df.head(3).to_csv(index=False))
    
    return "\n".join(summary)
