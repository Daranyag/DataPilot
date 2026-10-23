import pandas as pd

def eval_pandas():
    print("--- real_diamonds_54k.csv ---")
    df = pd.read_csv(r"testingg\test_data\real_diamonds_54k.csv")
    print("Rows and cols:", df.shape)
    print("Average price:", df["price"].mean())
    print("Average price by cut:\n", df.groupby("cut")["price"].mean())
    print("Most expensive cut on average:", df.groupby("cut")["price"].mean().idxmax())
    if "carat" in df.columns and "price" in df.columns:
        print("Correlation carat vs price:", df["carat"].corr(df["price"]))
    print("Missing values:\n", df.isnull().sum())
    
    print("\n--- real_iris.csv ---")
    df2 = pd.read_csv(r"testingg\test_data\real_iris.csv")
    print("Rows and cols:", df2.shape)
    print("Average of each column:\n", df2.mean(numeric_only=True))

    print("\n--- messy_customers.csv ---")
    df3 = pd.read_csv(r"testingg\test_data\messy_customers.csv")
    print("Missing values:\n", df3.isnull().sum())
    print("Duplicate rows:", df3.duplicated().sum())
    print("ID columns (unique per row):", [col for col in df3.columns if df3[col].nunique() == len(df3)])
    
if __name__ == "__main__":
    eval_pandas()
