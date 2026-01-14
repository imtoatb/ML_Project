import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')                                           #Only for the presentation

def harmonize_column_names(df_clean, df_synthetic):
    column_mapping = {
     
        'rotterdam_hyperandrogenie': 'rotterdam_hyperandrogenism',
        'rotterdam_ovaires_poly': 'rotterdam_poly_ovaries',
        'lh_fsh_ratio': 'fsh_lh_ratio',
        
     
        'Non-SOPK': 'Non-PCOS',
        'SOPK_atypique': 'Atypical_PCOS'
    }

    df_synthetic_clean = df_synthetic.rename(columns=column_mapping)
    

    if 'phenotype_rotterdam' in df_synthetic_clean.columns:                         #in addition of th phenotyê
        phenotype_mapping = {
            'Non-SOPK': 'Non-PCOS',
            'SOPK_atypique': 'Atypical_PCOS'
        }

        df_synthetic_clean['phenotype_rotterdam'] = df_synthetic_clean['phenotype_rotterdam'].replace(phenotype_mapping)
    return df_synthetic_clean

def align_columns_structure(df_clean, df_synthetic):
    expected_columns = [
        'pcos', 'age', 'weight_kg', 'height_cm', 'menstrual_regularity', 'cycle_length_days',
        'hair_growth_final', 'acne_score', 'fsh', 'lh', 'amh', 'insulin_resistance',
        'follicle_count_left', 'follicle_count_right', 'tsh', 'prolactin', 'progesterone',
        'blood_glucose', 'vitamin_d3', 'waist_hip_ratio', 'bmi', 'fsh_lh_ratio',
        'testosterone', 'fasting_insulin', 'homa_ir', 'hyperandrogenism_score',
        'total_follicles', 'ovarian_pcos_score', 'metabolic_risk_score', 'phenotype_rotterdam',
        'rotterdam_hyperandrogenism', 'rotterdam_dysovulation', 'rotterdam_poly_ovaries'
    ]
    
 
    clean_missing = set(expected_columns) - set(df_clean.columns)
    synthetic_missing = set(expected_columns) - set(df_synthetic.columns)
    
    if clean_missing:
        print(f"Missing columns in clean dataset: {clean_missing}")
    if synthetic_missing:
        print(f"Missing columns in synthetic dataset: {synthetic_missing} \n")
    
  
    for col in expected_columns:
        if col not in df_clean.columns:
            df_clean[col] = np.nan
            print(f"Column {col} added to clean dataset")
        if col not in df_synthetic.columns:
            df_synthetic[col] = np.nan
            print(f"Column {col} added to synthetic dataset\n")
    
 
    df_clean = df_clean[expected_columns]
    df_synthetic = df_synthetic[expected_columns]
    

    return df_clean, df_synthetic

def convert_menstrual_regularity_values(df):
    if 'menstrual_regularity' in df.columns:

        if df['menstrual_regularity'].dtype == object:
            regularity_map = {'high': 0, 'low': 1, 'haut': 0, 'bas': 1}
            df['menstrual_regularity'] = df['menstrual_regularity'].map(regularity_map)
            converted = df['menstrual_regularity'].notna().sum()                                #sum non missing val
            #print(f"   {converted} values converted\n")
    
    return df

def ensure_numeric_columns(df):

    numeric_columns = [
        'age', 'weight_kg', 'height_cm', 'cycle_length_days', 'hair_growth_final', 
        'acne_score', 'fsh', 'lh', 'amh', 'insulin_resistance', 'follicle_count_left',
        'follicle_count_right', 'tsh', 'prolactin', 'progesterone', 'blood_glucose',
        'vitamin_d3', 'waist_hip_ratio', 'bmi', 'fsh_lh_ratio', 'testosterone',
        'fasting_insulin', 'homa_ir', 'hyperandrogenism_score', 'total_follicles',
        'ovarian_pcos_score', 'metabolic_risk_score'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            before_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after_dtype = df[col].dtype
            nan_count = df[col].isna().sum()
            
            if nan_count > 0:
                print(f"{col}: {nan_count} values converted to NaN\n")
            elif before_dtype != after_dtype:
                print(f"{col}: converted to numeric\n")
    
    return df

def merge_datasets(df_clean, df_synthetic):
 

    df_clean['data_source'] = 'original_cleaned'
    df_synthetic['data_source'] = 'synthetic'
 
    df_merged = pd.concat([df_clean, df_synthetic], ignore_index=True)
    
  
    print(f"Clean dataset: {len(df_clean)} rows")
    print(f"Synthetic dataset: {len(df_synthetic)} rows")
    print(f"Merged dataset: {len(df_merged)} rows\n")
    
    return df_merged

def validate_merged_dataset(df):
    
    print(f"Dataset size: {df.shape}")
    print(f"Data types:")
    print(df.dtypes.value_counts())
    print("\n")
    
    if 'pcos' in df.columns:
        pcos_dist = df['pcos'].value_counts()
  
        for val, count in pcos_dist.items():
            percentage = count / len(df) * 100
            print(f"   PCOS={val}: {count} ({percentage:.1f}%)")
    

    if 'phenotype_rotterdam' in df.columns:
        pheno_dist = df['phenotype_rotterdam'].value_counts()

        for pheno, count in pheno_dist.items():
            percentage = count / len(df) * 100
            print(f"   {pheno}: {count} ({percentage:.1f}%)")
    print("\n")
    

    medical_ranges = {
        'age': (18, 47),
        'bmi': (16, 40),
        'testosterone': (15, 150),
        'total_follicles': (2, 50)
    }
    
    for col, (min_val, max_val) in medical_ranges.items():
        if col in df.columns:
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range > 0:
                print(f"{col}: {out_of_range} values out of range ({min_val}-{max_val})\n")
            else:
                print(f"{col}: all values in range ({min_val}-{max_val})\n")
    
    # Source consistency
    if 'data_source' in df.columns:

        source_analysis = df.groupby('data_source').agg({
            'pcos': 'mean',
            'age': 'mean',
            'bmi': 'mean',
            'testosterone': 'mean'
        }).round(2)
        print(source_analysis)

def save_final_dataset(df, filename='Final_Merged_Dataset.csv'):
    if 'data_source' in df.columns:
        df_to_save = df.drop('data_source', axis=1)
    else:
        df_to_save = df
    
    df_to_save.to_csv(filename, index=False)
    print(f" {filename}")
    print(f"{df_to_save.shape}")
    
    return df_to_save

def main_fusion_pipeline():


    df_clean = pd.read_csv('og_dataset_cleaned.csv')
    df_synthetic = pd.read_csv('Synthetic_Dataset_Cleaned.csv')
    
    print(f"   Clean dataset: {df_clean.shape}")
    print(f"   Synthetic dataset: {df_synthetic.shape}\n")
    
    df_synthetic_harmonized = harmonize_column_names(df_clean, df_synthetic)

    df_synthetic_converted = convert_menstrual_regularity_values(df_synthetic_harmonized)
    

    df_clean_aligned, df_synthetic_aligned = align_columns_structure(df_clean, df_synthetic_converted)

    df_clean_final = ensure_numeric_columns(df_clean_aligned)
    df_synthetic_final = ensure_numeric_columns(df_synthetic_aligned)
    df_merged = merge_datasets(df_clean_final, df_synthetic_final)
    validate_merged_dataset(df_merged)
    

    final_df = save_final_dataset(df_merged, 'final_merged_dataset.csv')
    

    return final_df
    

final_dataset = main_fusion_pipeline()

if final_dataset is not None:

    print(final_dataset.head())
    print(final_dataset['phenotype_rotterdam'].value_counts())