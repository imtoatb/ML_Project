import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_column_names(df):

    column_mapping = {
        'Sl. No': 'id',
        'Patient File No.': 'patient_id',
        'PCOS (Y/N)': 'pcos',
        'Age (yrs)': 'age',
        'Weight (Kg)': 'weight_kg',
        'Height(Cm)': 'height_cm',
        'Height(Cm) ': 'height_cm',
        'BMI': 'bmi',
        'Blood Group': 'blood_group',
        'Pulse rate(bpm)': 'pulse_rate',
        'RR (breaths/min)': 'respiratory_rate',
        'BP _Systolic (mmHg)': 'bp_systolic',
        'BP _Diastolic (mmHg)': 'bp_diastolic',
        'Hb(g/dl)': 'hemoglobin',
        'I beta-HCG(mIU/mL)': 'beta_hcg_1',
        'II  beta-HCG(mIU/mL)': 'beta_hcg_2',
        'FSH(mIU/mL)': 'fsh',
        'LH(mIU/mL)': 'lh',
        'FSH/LH': 'fsh_lh_ratio',
        'TSH (mIU/L)': 'tsh',
        'AMH(ng/mL)': 'amh',
        'PRL(ng/mL)': 'prolactin',
        'PRG(ng/mL)': 'progesterone',
        'Vit D3 (ng/mL)': 'vitamin_d3',
        'RBS(mg/dl)': 'blood_glucose',
        'Cycle(R/I)': 'menstrual_regularity',
        'Cycle length(days)': 'cycle_length_days',
        'Marraige Status (Yrs)': 'marriage_years',
        'Pregnant(Y/N)': 'ever_pregnant',
        'No. of abortions': 'abortions_count',
        'Hip(inch)': 'hip_inch',
        'Waist(inch)': 'waist_inch',
        'Waist:Hip Ratio': 'waist_hip_ratio',
        'Weight gain(Y/N)': 'weight_gain',
        'hair growth(Y/N)': 'hair_growth',
        'Skin darkening (Y/N)': 'skin_darkening',
        'Hair loss(Y/N)': 'hair_loss',
        'Pimples(Y/N)': 'pimples',
        'Fast food (Y/N)': 'fast_food',
        'Reg.Exercise(Y/N)': 'regular_exercise',
        'Follicle No. (L)': 'follicle_count_left',
        'Follicle No. (R)': 'follicle_count_right',
        'Avg. F size (L) (mm)': 'follicle_size_left_mm',
        'Avg. F size (R) (mm)': 'follicle_size_right_mm',
        'Endometrium (mm)': 'endometrium_thickness_mm'
    }
    
    
    df_clean = df.rename(columns=column_mapping)
    return df_clean



def convert_numeric_columns(df):
    numeric_columns = [
        'age', 'weight_kg', 'height_cm', 'bmi', 'pulse_rate', 'respiratory_rate',
        'bp_systolic', 'bp_diastolic', 'hemoglobin', 'beta_hcg_1', 'beta_hcg_2',
        'fsh', 'lh', 'fsh_lh_ratio', 'tsh', 'amh', 'prolactin', 'progesterone',
        'vitamin_d3', 'blood_glucose', 'cycle_length_days', 'marriage_years',
        'abortions_count', 'hip_inch', 'waist_inch', 'waist_hip_ratio',
        'follicle_count_left', 'follicle_count_right', 'follicle_size_left_mm',
        'follicle_size_right_mm', 'endometrium_thickness_mm'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
      
            df[col] = pd.to_numeric(df[col], errors='coerce')                   #To nb, If error, transform in NaN
    return df





def convert_to_binary(value):
    string_value = str(value)
    cleaned_value = string_value.strip()                #space
    normalized_value = cleaned_value.upper()            #Maj
   
    true_values = ['Y', 'YES', '1', 'TRUE']
    

    if normalized_value in true_values:
        return 1
    else:
        return 0


def convert_binary_variables(df):
    binary_variables = [
        'pcos', 'ever_pregnant', 'weight_gain', 'hair_growth',
        'skin_darkening', 'hair_loss', 'pimples', 'fast_food', 'regular_exercise'
    ]
    
    for col in binary_variables:
        if col in df.columns:
            df[col] = df[col].astype(str)                               #transform type -> string
            df[col] = df[col].apply(convert_to_binary)
    
    return df



def convert_menstrual_regularity(df):

    if 'menstrual_regularity' not in df.columns:
        return df
  
    df['menstrual_regularity'] = df['menstrual_regularity'].astype(str)
    
    regularity_map = {
        'Regular': 0, 'regular': 0, 'R': 0, 'r': 0, 2:0,
        'Irregular': 1, 'irregular': 1, 'I': 1, 'i': 1, 4:1
    }
    
    df['menstrual_regularity'] = df['menstrual_regularity'].map(
        lambda x: regularity_map.get(str(x).strip(), np.nan)
    )

    df['menstrual_regularity'] = pd.to_numeric(df['menstrual_regularity'], errors='coerce').fillna(0).astype(int)       #NaN -> 0
    
    return df

def correct_menstrual_cycle(df):
    if 'cycle_length_days' not in df.columns:
        return df

    df['cycle_length_days'] = pd.to_numeric(df['cycle_length_days'], errors='coerce')
    
    #day → cycle
    menstrual_bleeding_mask = (df['cycle_length_days'] >= 1) & (df['cycle_length_days'] <= 11)
    


    if menstrual_bleeding_mask.sum() > 0:
        
        n_to_convert = menstrual_bleeding_mask.sum()
   
        normal_cycles = np.random.randint(21, 36, size=n_to_convert)
        
     
        n_irregular = int(0.3 * n_to_convert)
        irregular_indices = np.random.choice(n_to_convert, size=n_irregular, replace=False)         #replace = can be selesct multiple time
    
        n_short = n_irregular // 2
        short_cycles = np.random.randint(15, 21, size=n_short)
        long_cycles = np.random.randint(36, 46, size=n_irregular - n_short)
        
      
        all_indices = df[menstrual_bleeding_mask].index
        df.loc[all_indices, 'cycle_length_days'] = normal_cycles                                    #Will use the random.randint
        
        if n_short > 0:
            short_indices = all_indices[irregular_indices[:n_short]]
            df.loc[short_indices, 'cycle_length_days'] = short_cycles
            
        if n_irregular - n_short > 0:
            long_indices = all_indices[irregular_indices[n_short:]]
            df.loc[long_indices, 'cycle_length_days'] = long_cycles
    

    #re-verif
    if 'menstrual_regularity' in df.columns:
        irregular_cycle_mask = (df['cycle_length_days'] < 21) | (df['cycle_length_days'] > 35)
        df.loc[irregular_cycle_mask, 'menstrual_regularity'] = 1                                                #loc =select a label
        
        regular_cycle_mask = (df['cycle_length_days'] >= 21) & (df['cycle_length_days'] <= 35)
        df.loc[regular_cycle_mask, 'menstrual_regularity'] = 0
    
    return df

def unify_skin_symptoms(df):

    if 'pimples' in df.columns:
        df['acne_score'] = df['pimples'].fillna(0)                  #rename
       
    
    if 'hair_growth' in df.columns:
        df['hair_growth_final'] = df['hair_growth']
    
    return df



def calculate_derived_variables(df):
    if 'bmi' not in df.columns and 'weight_kg' in df.columns and 'height_cm' in df.columns:
        df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
      
    
    
    if 'fsh_lh_ratio' not in df.columns and 'fsh' in df.columns and 'lh' in df.columns:
        df['fsh'] = pd.to_numeric(df['fsh'], errors='coerce')
        df['lh'] = pd.to_numeric(df['lh'], errors='coerce')
        df['fsh_lh_ratio'] = df['lh'] / df['fsh']
        df['fsh_lh_ratio'] = df['fsh_lh_ratio'].replace([np.inf, -np.inf], np.nan)          # if lh/0 = inf -> NaN

    if 'follicle_count_left' in df.columns and 'follicle_count_right' in df.columns:
        df['follicle_count_left'] = pd.to_numeric(df['follicle_count_left'], errors='coerce')
        df['follicle_count_right'] = pd.to_numeric(df['follicle_count_right'], errors='coerce')
        
        df['total_follicles'] = df['follicle_count_left'] + df['follicle_count_right']
       

    if 'acne_score' in df.columns and 'hair_growth_final' in df.columns:
        df['hyperandrogenism_score'] = df['acne_score'] + df['hair_growth_final']
       
    
    return df



def add_missing_variables_medical(df):
    n_patients = len(df)
    if 'age' not in df.columns or df['age'].isna().any():
        df['age'] = np.random.randint(18, 48, n_patients)
  
    
    if 'testosterone' not in df.columns or df['testosterone'].isna().any():
       
        testosterone_values = np.zeros(n_patients)
        
        if 'pcos' in df.columns:
            non_pcos_mask = df['pcos'] == 0
            if non_pcos_mask.sum() > 0:
                base_testo = np.random.normal(30, 8, non_pcos_mask.sum())                       #Mean of distr, st deviat°, size
                testosterone_values[non_pcos_mask] = np.clip(base_testo, 15, 45)                #clip = range
               
            pcos_mask = df['pcos'] == 1
            if pcos_mask.sum() > 0:
                pcos_indices = np.where(pcos_mask)[0]                                           #All PCOS = 0
                
                n_high = int(0.70 * len(pcos_indices))
                high_indices = pcos_indices[:n_high]
                testosterone_values[high_indices] = np.random.gamma(3, 15, n_high) + 40
                testosterone_values[high_indices] = np.clip(testosterone_values[high_indices], 60, 150)
                
                n_moderate = int(0.25 * len(pcos_indices))
                moderate_indices = pcos_indices[n_high:n_high + n_moderate]
                testosterone_values[moderate_indices] = np.random.normal(55, 3, n_moderate)
                testosterone_values[moderate_indices] = np.clip(testosterone_values[moderate_indices], 50, 60)
                
                n_normal = len(pcos_indices) - n_high - n_moderate
                if n_normal > 0:
                    normal_indices = pcos_indices[n_high + n_moderate:]
                    testosterone_values[normal_indices] = np.random.normal(35, 6, n_normal)
                    testosterone_values[normal_indices] = np.clip(testosterone_values[normal_indices], 25, 45)
       
        else:
            testosterone_values = np.random.gamma(2, 15, n_patients) + 20
            testosterone_values = np.clip(testosterone_values, 15, 150)
        
        df['testosterone'] = np.round(testosterone_values, 1)
  
    if 'menstrual_regularity' not in df.columns or df['menstrual_regularity'].isna().any():                 #isna = is NaN -> .any = if any Nan
        regularity_values = np.zeros(n_patients)
        
        if 'pcos' in df.columns:
            non_pcos_mask = df['pcos'] == 0
            if non_pcos_mask.sum() > 0:
                regularity_values[non_pcos_mask] = np.random.choice([0, 1], non_pcos_mask.sum(), p=[0.95, 0.05])
            
           
            pcos_mask = df['pcos'] == 1
            if pcos_mask.sum() > 0:
                regularity_values[pcos_mask] = np.random.choice([0, 1], pcos_mask.sum(), p=[0.10, 0.90])
        else:
            regularity_values = np.random.choice([0, 1], n_patients, p=[0.80, 0.20])            #non PCOS -> 80%, Pcos -> 20%
        
        df['menstrual_regularity'] = regularity_values.astype(int)
     
    
   
    medical_variables = {
        'fsh': {'non_pcos': (3, 10, 'lognormal', 1.2, 0.3), 'pcos': (2, 8, 'lognormal', 1.0, 0.3)},
        'lh': {'non_pcos': (3, 10, 'lognormal', 1.2, 0.3), 'pcos': (8, 25, 'lognormal', 2.3, 0.4)},
        'amh': {'non_pcos': (1, 4, 'gamma', 2, 1), 'pcos': (4, 15, 'gamma', 3, 2)},
        'tsh': {'non_pcos': (0.4, 2.5, 'lognormal', 0, 0.5), 'pcos': (0.4, 4.0, 'lognormal', 0.5, 0.6)},
        'prolactin': {'non_pcos': (5, 25, 'gamma', 2, 6), 'pcos': (10, 40, 'gamma', 2, 8)},
        'vitamin_d3': {'non_pcos': (30, 80, 'normal', 55, 15), 'pcos': (10, 25, 'normal', 17, 5)},
        'fasting_insulin': {'non_pcos': (2, 15, 'gamma', 2, 3), 'pcos': (10, 30, 'gamma', 2, 6)}
    }
    
    #pcos/nonpcos (min, max, distributn, param1, param2)

    #loop through each variables
    for var, ranges in medical_variables.items():
        if var not in df.columns or df[var].isna().any():
            values = np.zeros(n_patients)
            
            if 'pcos' in df.columns:
                non_pcos_mask = df['pcos'] == 0
                pcos_mask = df['pcos'] == 1
                
                if non_pcos_mask.sum() > 0:
                    if ranges['non_pcos'][2] == 'lognormal':
                        values[non_pcos_mask] = np.random.lognormal(ranges['non_pcos'][3], ranges['non_pcos'][4], non_pcos_mask.sum())
                    elif ranges['non_pcos'][2] == 'gamma':
                        values[non_pcos_mask] = np.random.gamma(ranges['non_pcos'][3], ranges['non_pcos'][4], non_pcos_mask.sum())
                    else:  # normal
                        values[non_pcos_mask] = np.random.normal(ranges['non_pcos'][3], ranges['non_pcos'][4], non_pcos_mask.sum())
                    
                    values[non_pcos_mask] = np.clip(values[non_pcos_mask], ranges['non_pcos'][0], ranges['non_pcos'][1])
                
                if pcos_mask.sum() > 0:
                    if ranges['pcos'][2] == 'lognormal':
                        values[pcos_mask] = np.random.lognormal(ranges['pcos'][3], ranges['pcos'][4], pcos_mask.sum())
                    elif ranges['pcos'][2] == 'gamma':
                        values[pcos_mask] = np.random.gamma(ranges['pcos'][3], ranges['pcos'][4], pcos_mask.sum())
                    else:  # normal
                        values[pcos_mask] = np.random.normal(ranges['pcos'][3], ranges['pcos'][4], pcos_mask.sum())
                    
                    values[pcos_mask] = np.clip(values[pcos_mask], ranges['pcos'][0], ranges['pcos'][1])
            
            
            else:
                if ranges['non_pcos'][2] == 'lognormal':
                    values = np.random.lognormal(ranges['non_pcos'][3], ranges['non_pcos'][4], n_patients)
                elif ranges['non_pcos'][2] == 'gamma':
                    values = np.random.gamma(ranges['non_pcos'][3], ranges['non_pcos'][4], n_patients)
                else:  # normal
                    values = np.random.normal(ranges['non_pcos'][3], ranges['non_pcos'][4], n_patients)
                
                
                #Human range (non regarding sopk or not)
                min_val = min(ranges['non_pcos'][0], ranges['pcos'][0])
                max_val = max(ranges['non_pcos'][1], ranges['pcos'][1])
                values = np.clip(values, min_val, max_val)
            
            df[var] = np.round(values, 2)
         
    
   
    if 'homa_ir' not in df.columns and 'fasting_insulin' in df.columns and 'blood_glucose' in df.columns:
        df['homa_ir'] = (df['fasting_insulin'] * df['blood_glucose']) / 405
        df['homa_ir'] = np.round(df['homa_ir'], 2)

    if 'insulin_resistance' not in df.columns and 'homa_ir' in df.columns:
        df['insulin_resistance'] = np.zeros(n_patients)
        low_ir_mask = df['homa_ir'] < 2.5
        high_ir_mask = df['homa_ir'] >= 2.5
        
        if low_ir_mask.sum() > 0:
            df.loc[low_ir_mask, 'insulin_resistance'] = np.random.poisson(1, low_ir_mask.sum())
        if high_ir_mask.sum() > 0:
            df.loc[high_ir_mask, 'insulin_resistance'] = np.random.poisson(2.5, high_ir_mask.sum()) + 1
        
        df['insulin_resistance'] = np.clip(df['insulin_resistance'], 0, 4).astype(int)
     
    
    return df



def calculate_composite_scores(df):
    numeric_cols = ['amh', 'total_follicles', 'fsh_lh_ratio', 'bmi', 'waist_hip_ratio', 'homa_ir', 'insulin_resistance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    

    if all(col in df.columns for col in ['amh', 'total_follicles', 'fsh_lh_ratio']):
        ovarian_score = np.zeros(len(df))
        ovarian_score += np.where(df['amh'] > 7, 2, np.where(df['amh'] > 4, 1, 0))              #for each patient, if ...> 7 then +=2, if ...>4 then +=1 else 0
        ovarian_score += np.where(df['total_follicles'] > 25, 2, np.where(df['total_follicles'] > 20, 1, 0))
        ovarian_score += np.where(df['fsh_lh_ratio'] > 2, 1, 0)
        df['ovarian_pcos_score'] = ovarian_score.astype(int)
    
    
  
    if all(col in df.columns for col in ['insulin_resistance', 'bmi', 'waist_hip_ratio', 'homa_ir']):
        metabolic_score = np.zeros(len(df))
        metabolic_score += df['insulin_resistance']
        metabolic_score += np.where(df['bmi'] > 30, 2, np.where(df['bmi'] > 25, 1, 0))
        metabolic_score += np.where(df['waist_hip_ratio'] > 0.85, 1, 0)
        metabolic_score += np.where(df['homa_ir'] > 2.5, 1, 0)
        df['metabolic_risk_score'] = metabolic_score.astype(int)
      
    return df

def apply_medical_criteria(df):
    required_cols = ['testosterone', 'hair_growth_final', 'acne_score', 'hyperandrogenism_score', 
                    'menstrual_regularity', 'cycle_length_days', 'progesterone', 
                    'total_follicles', 'follicle_count_left', 'follicle_count_right', 'ovarian_pcos_score']
    
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
   
    hyperandrogenism = ((df['testosterone'] >= 45) | (df['hair_growth_final'] >= 2.7) | (df['acne_score'] >= 3) | (df['hyperandrogenism_score'] >= 3)).astype(int)
    dysovulation = ((df['menstrual_regularity'] == 1) | (df['cycle_length_days'] < 21) | (df['cycle_length_days'] > 35) |  (df['progesterone'] < 3)).astype(int)
    poly_ovaries = ((df['total_follicles'] >= 20) | (df['follicle_count_left'] >= 12) | (df['follicle_count_right'] >= 12) | (df['ovarian_pcos_score'] >= 3)).astype(int)
    
    pcos_diagnosis = (hyperandrogenism + dysovulation + poly_ovaries >= 2).astype(int)
    
    return pcos_diagnosis, hyperandrogenism, dysovulation, poly_ovaries






def determine_phenotypes_medical(df):
    phenotypes = []
    
    for _, row in df.iterrows():
        if row['pcos'] == 1:
            hyper = row['rotterdam_hyperandrogenism']
            dysov = row['rotterdam_dysovulation']
            ovarian = row['rotterdam_poly_ovaries']
            
          
            if hyper == 1 and dysov == 1 and ovarian == 1:
                phenotypes.append('Phenotype_a')  # Complete
            elif hyper == 1 and dysov == 1 and ovarian == 0:
                phenotypes.append('Phenotype_b')  # Hyperandrogenism + Anovulation
            elif hyper == 1 and dysov == 0 and ovarian == 1:
                phenotypes.append('Phenotype_c')  # Hyperandrogenism + PCO
            elif hyper == 0 and dysov == 1 and ovarian == 1:
                phenotypes.append('Phenotype_d')  # Anovulation + PCO
            else:
                phenotypes.append('Atypical_PCOS')
        
        else:
            phenotypes.append('Non-PCOS')
    

    df['phenotype_rotterdam'] = phenotypes
    

    distribution = df['phenotype_rotterdam'].value_counts()
    """print("PHENOTYPE DISTRIBUTION:")
    for pheno, count in distribution.items():
        percentage = count / len(df) * 100
        print(f"{pheno}: {count} patients ({percentage:.1f}%)")
    """
    #print("\n")
    return df

def get_final_columns():
    return [
        'pcos', 'age', 'weight_kg', 'height_cm', 'menstrual_regularity', 'cycle_length_days',
        'hair_growth_final', 'acne_score', 'fsh', 'lh', 'amh', 'insulin_resistance',
        'follicle_count_left', 'follicle_count_right', 'tsh', 'prolactin', 'progesterone',
        'blood_glucose', 'vitamin_d3', 'waist_hip_ratio', 'bmi', 'fsh_lh_ratio',
        'testosterone', 'fasting_insulin', 'homa_ir', 'hyperandrogenism_score',
        'total_follicles', 'ovarian_pcos_score', 'metabolic_risk_score', 'phenotype_rotterdam',
        'rotterdam_hyperandrogenism', 'rotterdam_dysovulation', 'rotterdam_poly_ovaries'
    ]

def prepare_final_dataset(df):
    final_columns = get_final_columns()
    available_columns = [col for col in final_columns if col in df.columns]
    
    df_final = df[available_columns].copy()
    return df_final


#To visualize
def validate_medical_coherence(df):
    pcos_mask = df['pcos'] == 1
    non_pcos_mask = df['pcos'] == 0
    
    print(f"Dataset overview: {len(df)} patients")
    print(f"PCOS: {pcos_mask.sum()} ({pcos_mask.sum()/len(df)*100:.1f}%)")
    print(f"Non-PCOS: {non_pcos_mask.sum()} ({non_pcos_mask.sum()/len(df)*100:.1f}%)")
    
    if 'testosterone' in df.columns:
        testo_pcos = df.loc[pcos_mask, 'testosterone'].mean()
        testo_non_pcos = df.loc[non_pcos_mask, 'testosterone'].mean()
        #print(f"Testosterone - PCOS: {testo_pcos:.1f} ng/dL, Non-PCOS: {testo_non_pcos:.1f} ng/dL")
    
    if 'menstrual_regularity' in df.columns:
        irregular_pcos = (df.loc[pcos_mask, 'menstrual_regularity'] == 1).mean() * 100
        irregular_non_pcos = (df.loc[non_pcos_mask, 'menstrual_regularity'] == 1).mean() * 100
        #print(f"Irregular cycles - PCOS: {irregular_pcos:.1f}%, Non-PCOS: {irregular_non_pcos:.1f}%")
    
   
    if 'total_follicles' in df.columns:
        follicles_pcos = df.loc[pcos_mask, 'total_follicles'].mean()
        follicles_non_pcos = df.loc[non_pcos_mask, 'total_follicles'].mean()
        #print(f"Total follicles - PCOS: {follicles_pcos:.1f}, Non-PCOS: {follicles_non_pcos:.1f}")


    if 'homa_ir' in df.columns:
        homair_pcos = df.loc[pcos_mask, 'homa_ir'].mean()
        homair_non_pcos = df.loc[non_pcos_mask, 'homa_ir'].mean()
        #print(f"HOMA-IR - PCOS: {homair_pcos:.2f}, Non-PCOS: {homair_non_pcos:.2f}\n")




def process_complete_dataset_medical(source_file):
    df = pd.read_csv(source_file)

    df = clean_column_names(df)
    df = convert_numeric_columns(df)
    df = convert_binary_variables(df)
    df = convert_menstrual_regularity(df)
    
    df = unify_skin_symptoms(df)
    df = calculate_derived_variables(df)
    df = correct_menstrual_cycle(df)
    df = add_missing_variables_medical(df)

    df = calculate_composite_scores(df)
    
 
    pcos_diagnosis, hyper, dysov, poly = apply_medical_criteria(df)
    df['pcos'] = pcos_diagnosis
    df['rotterdam_hyperandrogenism'] = hyper
    df['rotterdam_dysovulation'] = dysov
    df['rotterdam_poly_ovaries'] = poly
    
    df = determine_phenotypes_medical(df)

    df_final = prepare_final_dataset(df)
    
 
    validate_medical_coherence(df_final)
    return df_final



df_final = process_complete_dataset_medical('./Dataset/PCOS_data.csv')

#
output_file = 'og_dataset_cleaned.csv'
df_final.to_csv(output_file, index=False)


print(f"Shape: {df_final.shape}")
print(f"PCOS distribution:")
print(df_final['pcos'].value_counts())
print(f"\nPhenotype distribution:")
print(df_final['phenotype_rotterdam'].value_counts())