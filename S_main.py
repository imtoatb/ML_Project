import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns




def medical_features(df):

    clinical_features = [
        'age', 'weight_kg', 'height_cm', 'bmi', 'testosterone', 'homa_ir', 
        'amh', 'total_follicles', 'menstrual_regularity', 'cycle_length_days',
        'hair_growth_final', 'acne_score', 'fsh', 'lh', 'fsh_lh_ratio',
        'insulin_resistance', 'follicle_count_left', 'follicle_count_right',
        'tsh', 'prolactin', 'progesterone', 'blood_glucose', 'vitamin_d3',
        'waist_hip_ratio', 'fasting_insulin'
    ]
    
    available_features = [f for f in clinical_features if f in df.columns]
    return df[available_features + ['pcos']]                                                            #+ column pcos to redo the calculations




def id_sopk_patients(X_train, X_test, y_train, y_test, feature_names):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),                                                  #true for roc [0,1,0] -> [0.2,0.7,0.4]
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)                       #max_iter=1000 to ensure convergence
    }
    
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))                                                    #4 graphs on the same page (2x2), size 15x12 inches
    axes = axes.ravel()                                                                                 #2D -> 1D
    


    for i, (name, model) in enumerate(models.items()):
        if name in ['SVM', 'Logistic Regression']:                                                      #Need to be standardized because going to compare the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)                                              #proba that patient n has pcos (0.00 - 1.00)
            X_test_scaled = scaler.transform(X_test)                                                    #Answer 0 or 1
            
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]                                          #[[proba to be non-pcos, prob to be pcos], ...] take only the prob to be pcos
            y_pred = model.predict(X_test_scaled)


        else:                                                                                           #Don't neeed to scale bcs features are independent + it's tree
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        



        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)                                                        #do my model knows how to do the diff between a sopk and a non sopk (Auc goal > 0.8, capacity to distinguish between classes)
        fpr, tpr, _ = roc_curve(y_test, y_proba)                                                        #fpr, tpr, thresholds
        





        axes[i].plot(fpr, tpr, label=f'{name} (AUC = {auc_roc:.6f})')
        axes[i].plot((0, 1), (0, 1), linestyle='--')                                                    #random performance line as reference
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'ROC - {name}   Accuracy: {accuracy:.6f}')
        axes[i].legend()
        fig.tight_layout()
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"{name:<20} Accuracy: {accuracy:.6f}, AUC: {auc_roc:.6f}")
    
        
        #cm = add_confusion_matrix_binary(y_test, y_pred, name)
    
    fig.tight_layout()
    plt.show()
    



    best_model_name = max(results.keys(), key=lambda x: results[x]['auc_roc'] if results[x] else 0)
    best_result = results[best_model_name]
    


    if hasattr(best_result['model'], 'feature_importances_'):                                       #hasattr = verify model has attribute feature_importances_
        feature_importance = pd.DataFrame({'feature': feature_names,'importance': best_result['model'].feature_importances_}).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(10)                                                  #To get the 10 best features
        plt.barh(top_features['feature'], top_features['importance'])                               #horizontal bar diagram
        plt.xlabel('Importance')
        plt.title(f'10 Best Features of {best_model_name}')
        plt.gca().invert_yaxis()                                                                    #Inversed y
        #plt.show()
    
    print("\n")

    
        

    return results, best_model_name, best_result





def create_pcos_subtype_labels(df):
    required_cols = ['rotterdam_hyperandrogenism', 'rotterdam_dysovulation', 'rotterdam_poly_ovaries']
    
    if all(col in df.columns for col in required_cols):
        conditions = [
            (df[required_cols[0]] == 1) & (df[required_cols[1]] == 1) & (df[required_cols[2]] == 1),
            (df[required_cols[0]] == 1) & (df[required_cols[1]] == 1) & (df[required_cols[2]] == 0),
            (df[required_cols[0]] == 1) & (df[required_cols[1]] == 0) & (df[required_cols[2]] == 1),
            (df[required_cols[0]] == 0) & (df[required_cols[1]] == 1) & (df[required_cols[2]] == 1),
        ]
        
        choices = ['A', 'B', 'C', 'D']
        df['pcos_subtype'] = np.select(conditions, choices, default='Non-PCOS')
        return df
    


    else:
        return None




def visualize_subtype_distribution(df):
    if 'pcos_subtype' not in df.columns:
        return None
    
    subtype_counts = df['pcos_subtype'].value_counts()
    colors = {'A': 'red', 'B': 'orange', 'C': 'green', 'D': 'blue', 'Non-PCOS': 'gray'}
    color_list = [colors.get(subtype, 'black') for subtype in subtype_counts.index]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(subtype_counts.index, subtype_counts.values, color=color_list)
    plt.xlabel('PCOS Subtype')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of PCOS Subtypes')
    #plt.show()
    



def put_subtype_label(df):
    df_with_subtypes = create_pcos_subtype_labels(df)
    if df_with_subtypes is None:
        return None, None, None                                                                 #return X, y, feature_name
    
    clinical_features = [
        'age', 'weight_kg', 'height_cm', 'bmi', 'testosterone', 'homa_ir', 
        'amh', 'total_follicles', 'menstrual_regularity', 'cycle_length_days',
        'hair_growth_final', 'acne_score', 'fsh', 'lh', 'fsh_lh_ratio',
        'insulin_resistance', 'follicle_count_left', 'follicle_count_right',
        'tsh', 'prolactin', 'progesterone', 'blood_glucose', 'vitamin_d3',
        'waist_hip_ratio', 'fasting_insulin'
    ]
    
    available_features = [f for f in clinical_features if f in df_with_subtypes.columns]
    
    X = df_with_subtypes[available_features]
    y = df_with_subtypes['pcos_subtype']
    
   
    label_encoder = LabelEncoder()                                                              #Transform Non,A,B,C,D en 0,1,2,3,4
    y_encoded = label_encoder.fit_transform(y)                                                  #column with only nb to tell the subtype
    
    return X, y_encoded, available_features, label_encoder






def compare_subtype_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': OneVsRestClassifier(SVC(probability=True, random_state=42)),                             #binary classification -> is it A or (Non,B,C,D), is it B or (...),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')    #is it C, is it D ...if no everywhere then Non
    }
    
    results = {}
    
    for name, model in models.items():
        if name in ['SVM', 'Logistic Regression']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"{name:<20} Accuracy: {accuracy:.6f}")
    
    
    best_model_name = max(results.keys(), key = lambda x: results[x]['accuracy'] if results[x] else 0)
    best_result = results[best_model_name]
    
    return results, best_model_name, best_result




def create_test_patient(feature_names):

    pcos_values = {
        'testosterone': 65.0,
        'homa_ir': 3.2,
        'bmi': 28.5,
        'amh': 6.8,
        'total_follicles': 22,
        'menstrual_regularity': 1,
        'age': 29,
        'weight_kg': 72,
        'height_cm': 165,
        'fsh_lh_ratio': 1.2,
        'hair_growth_final': 1.8,
        'acne_score': 1.5,
    }

    normal_values = {
        'testosterone': 25.0,     # Normal: 15-45 ng/dL
        'homa_ir': 1.2,           # Normal: <2.0
        'bmi': 22.0,              # Normal: 18.5-24.9
        'amh': 2.5,               # Normal pour âge: 1.0-4.0 ng/mL
        'total_follicles': 10,    # Normal: 5-15
        'menstrual_regularity': 0,
        'age': 28,
        'weight_kg': 60,
        'height_cm': 165,
        'fsh_lh_ratio': 1.0,      
        'hair_growth_final': 0.5, 
        'acne_score': 0.3,        
    }

    borderline_values = {
        'testosterone': 48.0,     
        'homa_ir': 2.5,           
        'bmi': 26.0,              
        'amh': 4.5,               
        'total_follicles': 18,    
        'menstrual_regularity': 0,
        'age': 30,
        'weight_kg': 68,
        'height_cm': 162,
        'fsh_lh_ratio': 1.5,      
        'hair_growth_final': 1.2, 
        'acne_score': 1.0,            
    }

    mild_pcos_values = {
        'testosterone': 55.0,     
        'homa_ir': 3.0,           
        'bmi': 27.0,              
        'amh': 5.5,               
        'total_follicles': 10,    
        'menstrual_regularity': 1,
        'age': 32,
        'weight_kg': 70,
        'height_cm': 160,
        'fsh_lh_ratio': 2.0,     
        'hair_growth_final': 1.2, 
        'acne_score': 1.3,         
    }

    mixed_values = {
        'testosterone': 28.0,      
        'homa_ir': 1.5,  
        'bmi': 23.0,    
        'amh': 6.0,     
        'total_follicles': 11,  
        'menstrual_regularity': 0,
        'age': 25,
        'weight_kg': 58,
        'height_cm': 168,
        'fsh_lh_ratio': 1.1,     
        'hair_growth_final': 0.4, 
        'acne_score': 0.5,
    }
    
    patient_data = []
    for feature in feature_names:
        if feature in mild_pcos_values:
            patient_data.append(mild_pcos_values[feature])
        else:
            patient_data.append(0.0)                                                    #if not indicate here, put 0.0 by default
    
    return patient_data






def predict_binary_pcos(results, best_model_name, feature_names, patient_data):
    best_result = results[best_model_name]
    model = best_result['model']
    
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    
    if best_model_name in ['SVM', 'Logistic Regression']:
        scaler = StandardScaler()
        patient_scaled = scaler.fit_transform(patient_df)
        pcos_prob = model.predict_proba(patient_scaled)[0, 1]
    

    else:
        pcos_prob = model.predict_proba(patient_df)[0, 1]
    
    pcos_diagnosis = pcos_prob > 0.5
    
    print(f"Probability PCOS: {pcos_prob:.3f}")
    print(f"Diagnosis: {'PCOS' if pcos_diagnosis else 'NON-PCOS'}")
    
    important_features = ['testosterone', 'homa_ir', 'bmi', 'amh', 'total_follicles']






def predict_subtype_for_patient(model, feature_names, patient_data, label_encoder):
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    
    if isinstance(model, (OneVsRestClassifier, LogisticRegression)):                                                                #Verify if model is of an object of type  (...)
        scaler = StandardScaler()
        patient_scaled = scaler.fit_transform(patient_df)
        predicted_encoded = model.predict(patient_scaled)[0]
    
    else:
        predicted_encoded = model.predict(patient_df)[0]
    
    predicted_subtype = label_encoder.inverse_transform([predicted_encoded])[0]                 #To reverse [0,1,2,3,4] -> [Non,A,B,C,D]
    
    print(f"\nPCOS Subtype Prediction:")
    print(f"Predicted subtype: {predicted_subtype}")
    
    return predicted_subtype




def add_confusion_matrix_binary(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-PCOS', 'PCOS'], yticklabels=['Non-PCOS', 'PCOS'])  #annot = nb annotation in the box, d for decimal
    plt.xlabel('Predictions')
    plt.ylabel('Reality')
    plt.title(f'Confusion Matrix {model_name} (Binary)')
    #plt.show()
    
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\n")
    return cm



def add_confusion_matrix_subtype(y_true, y_pred, label_encoder, model_name):
    class_names = label_encoder.classes_
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predictions')
    plt.ylabel('Réality')
    plt.title(f'Confusion Matrix {model_name} (subtype)')
    plt.tight_layout()
    #plt.show()
    
    print(f"\n{model_name}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return cm





#-----------------------------------------------------------------------------------------------------------------------------




df = pd.read_csv('final_merged_dataset.csv')
df_clean = medical_features(df)
X_binary = df_clean.drop('pcos', axis=1)                                                    #got rid of the pcos column beacause = answer (axis 1 = column)
y_binary = df_clean['pcos']                                                                 #put the pcos column here
feature_names = X_binary.columns.tolist()                                                   #To get the columns's names
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
print(f"Train: {X_train_bin.shape}, Test: {X_test_bin.shape} \n")


binary_results, best_binary_model, best_binary_result = id_sopk_patients(X_train_bin, X_test_bin, y_train_bin, y_test_bin, feature_names)


print("\n")
print(f"Best Model: {best_binary_model}")
print(f"Accuracy: {best_binary_result['accuracy']:.3f}")
print(f"AUC-ROC: {best_binary_result['auc_roc']:.3f}\n")

patient_data = create_test_patient(feature_names)
predict_binary_pcos(binary_results, best_binary_model, feature_names, patient_data)
print("\n")







X_subtype, y_subtype, subtype_features, label_encoder = put_subtype_label(df)

if X_subtype is not None:
    visualize_subtype_distribution(df)
    
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_subtype, y_subtype, test_size=0.2, random_state=42, stratify=y_subtype)
    print(f"\nTrain: {X_train_sub.shape}, Test: {X_test_sub.shape}")
  
    subtype_results, best_subtype_model, best_subtype_result = compare_subtype_models(X_train_sub, X_test_sub, y_train_sub, y_test_sub)
    

    for name, model_info in subtype_results.items():

        y_pred = model_info['predictions']
        #cm = add_confusion_matrix_subtype(y_test_sub, y_pred, label_encoder, name)



    patient_subtype_data = []
    for feature in subtype_features:
        if feature in feature_names:
            idx = feature_names.index(feature)
            patient_subtype_data.append(patient_data[idx])
        else:
            patient_subtype_data.append(0.0)
    
    predicted_subtype = predict_subtype_for_patient(best_subtype_result['model'], subtype_features, patient_subtype_data, label_encoder)
    
    print(f"\nSubtype Classification Results:")
    print(f"Best Model: {best_subtype_model}")
    print(f"Accuracy: {best_subtype_result['accuracy']:.3f}")
else:
    print("Problem")




