import pandas as pd
import os


def clean_sqli_data(input_path, output_path):
    """
    Cleans and prepares the SQL injection dataset.
    Implements the preprocessing steps described in Section 3.4.2 of the research paper:
      - Removes nulls and duplicates
      - Consolidates shifted label columns
      - Normalizes text to lowercase (SQL is case-insensitive)
      - Retains special characters (', --, ;, /*) which are critical for SQLi detection
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Load dataset
    df = pd.read_csv(input_path)

    # Function to find label across multiple columns due to CSV shifting
    def consolidate_label(row):
        for col in ['label', 'Unnamed: 2', 'Unnamed: 3']:
            val = str(row[col]).strip()
            # Check for various representations of 0 and 1
            if val in ['0', '1', '0.0', '1.0']:
                return int(float(val))
        return None

    print("Cleaning data...")
    df['label'] = df.apply(consolidate_label, axis=1)

    # Remove nulls and duplicates
    df_clean = df.dropna(subset=['text', 'label']).copy()
    df_clean['label'] = df_clean['label'].astype(int)

    # Lowercase normalization
    # SQL is case-insensitive: SELECT == select == Select
    df_clean['text'] = df_clean['text'].astype(str).str.lower()

    df_clean = df_clean[['text', 'label']].drop_duplicates()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_clean.to_csv(output_path, index=False)
    print(f"Success! Cleaned data saved to: {output_path}")
    print(f"Final count: {len(df_clean)} rows.")


if __name__ == "__main__":
    from src import config
    clean_sqli_data(str(config.DATA_RAW), str(config.DATA_PROCESSED))