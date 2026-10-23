import pandas as pd
import numpy as np

def generate_profile(df: pd.DataFrame) -> dict:
    """
    Computes a comprehensive data profile for the given DataFrame.
    Returns hard numbers to ensure the LLM doesn't hallucinate.
    """
    profile = {}
    
    # 1. Basic Shape
    profile['n_rows'] = len(df)
    profile['n_cols'] = len(df.columns)
    
    # 2. Duplicates
    profile['n_duplicates'] = int(df.duplicated().sum())
    
    # 3. Missing Values
    missing_counts = df.isna().sum()
    profile['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    profile['total_missing'] = int(missing_counts.sum())
    
    # 4. Column Types
    profile['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    profile['numeric_cols'] = list(df.select_dtypes(include=[np.number]).columns)
    profile['categorical_cols'] = list(df.select_dtypes(exclude=[np.number]).columns)
    
    # 5. Outliers (IQR Method) for numeric columns
    outliers = {}
    for col in profile['numeric_cols']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count values outside bounds
        outlier_count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        if outlier_count > 0:
            outliers[col] = outlier_count
            
    profile['outliers'] = outliers
    
    # 6. Basic Statistics for Numeric Cols
    if profile['numeric_cols']:
        stats = df[profile['numeric_cols']].describe().to_dict()
        profile['statistics'] = stats
    else:
        profile['statistics'] = {}
        
    # 7. Top Correlations
    profile['top_correlations'] = []
    if len(profile['numeric_cols']) > 1:
        corr_matrix = df[profile['numeric_cols']].corr().abs()
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find index of features with correlation greater than 0
        corr_pairs = upper.unstack().dropna().sort_values(ascending=False)
        
        # Take top 5 highest absolute correlations
        for (feat1, feat2), val in corr_pairs.head(5).items():
            if val > 0.3:  # Only report somewhat strong correlations
                profile['top_correlations'].append({
                    'feature1': feat1,
                    'feature2': feat2,
                    'correlation': round(val, 2)
                })
                
    return profile
