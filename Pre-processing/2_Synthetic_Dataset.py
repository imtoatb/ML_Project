import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
from sklearn.impute import SimpleImputer

# Chargement des données
df_combined = pd.read_csv('final_merged_dataset.csv')

print("=== ANALYSE INITIALE DES DONNÉES ===")
print(f"Shape du dataset: {df_combined.shape}")
print("\nColonnes disponibles:")
print(df_combined.columns.tolist())

# FEATURES OPTIMALES POUR LE SOUS-TYPAGE SOPK
clustering_features = [
    'bmi', 'testosterone', 'homa_ir', 'fsh_lh_ratio', 'amh',
    'waist_hip_ratio', 'total_follicles', 'hyperandrogenism_score',
    'age', 'insulin_resistance', 'menstrual_regularity'
]

print(f"\nFeatures sélectionnées pour le clustering: {clustering_features}")

# 1. PRÉPARATION DES DONNÉES
X = df_combined[clustering_features].copy()

print("\n=== ANALYSE DES VALEURS MANQUANTES ===")
print("Valeurs manquantes par feature:")
print(X.isnull().sum())

print("\n=== VALEURS UNIQUES DANS MENSTRUAL_REGULARITY ===")
print("Avant conversion:")
print(X['menstrual_regularity'].value_counts())

# Conversion robuste de menstrual_regularity
def convert_menstrual_regularity(val):
    if val in [0, '0', 'regular', 'high']:
        return 0
    elif val in [1, '1', 'irregular', 'low']:
        return 1
    else:
        return np.nan

X['menstrual_regularity'] = X['menstrual_regularity'].apply(convert_menstrual_regularity)

print("\nAprès conversion:")
print(X['menstrual_regularity'].value_counts(dropna=False))

# 2. IMPUTATION DES VALEURS MANQUANTES
print("\n=== IMPUTATION DES VALEURS MANQUANTES ===")

# Imputation séparée pour menstrual_regularity (catégorielle)
menstrual_na_count = X['menstrual_regularity'].isna().sum()
if menstrual_na_count > 0:
    print(f"Imputation de {menstrual_na_count} valeurs manquantes dans menstrual_regularity")
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X['menstrual_regularity'] = imputer_cat.fit_transform(X[['menstrual_regularity']])

# Imputation pour les autres variables numériques
numeric_features = [f for f in clustering_features if f != 'menstrual_regularity']
imputer_num = SimpleImputer(strategy='median')
X[numeric_features] = imputer_num.fit_transform(X[numeric_features])

print("Valeurs manquantes après imputation:")
print(X.isnull().sum())

# 3. STANDARDISATION DES DONNÉES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Vérification finale
print(f"\nShape après standardisation: {X_scaled.shape}")
print(f"NaN dans X_scaled: {np.isnan(X_scaled).sum()}")

# 4. RÉDUCTION DE DIMENSION POUR VISUALISATION
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA réussie! Variance expliquée: {pca.explained_variance_ratio_}")

# 5. DÉTERMINATION DU NOMBRE OPTIMAL DE CLUSTERS
print("\n=== DÉTERMINATION DU NOMBRE DE CLUSTERS ===")
inertia = []
silhouette_scores = []
k_range = range(2, 8)

plt.figure(figsize=(15, 5))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"K={k}, Score Silhouette: {silhouette_avg:.4f}")

# Graphique méthode du coude
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du Coude')
plt.grid(True)

# Graphique score de silhouette
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score Silhouette')
plt.title('Score de Silhouette')
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. CHOIX DU K OPTIMAL
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nK optimal sélectionné: {optimal_k} (meilleur score silhouette: {max(silhouette_scores):.4f})")

# 7. K-MEANS FINAL
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df_combined['cluster'] = cluster_labels
df_combined['cluster'] = df_combined['cluster'].astype(str)

# 8. VISUALISATION PCA
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title(f'Sous-typage SOPK - {optimal_k} Clusters')
plt.show()

# 9. ANALYSE DES CLUSTERS
print("\n=== ANALYSE DES SOUS-TYPES SOPK ===")
cluster_profiles = df_combined.groupby('cluster')[clustering_features].mean()

print("Caractéristiques moyennes par sous-type:")
print(cluster_profiles.round(3))

# DÉFINITION DE global_median (correction du bug)
global_median = cluster_profiles.median()

# Visualisation des profils de clusters
plt.figure(figsize=(14, 8))
for i, feature in enumerate(clustering_features[:6], 1):  # Premier 6 features
    plt.subplot(2, 3, i)
    for cluster in cluster_profiles.index:
        plt.bar(cluster, cluster_profiles.loc[cluster, feature], label=f'Cluster {cluster}')
    plt.title(f'{feature}')
    plt.xlabel('Cluster')
    plt.ylabel('Valeur moyenne')
plt.tight_layout()
plt.show()

# 10. INTERPRÉTATION MÉDICALE DES SOUS-TYPES
def interpret_pcos_subtypes(cluster_profiles):
    interpretations = {}
    global_median = cluster_profiles.median()
    
    for cluster_id in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster_id]
        
        # Définition des seuils basés sur la médiane globale
        high_ir = profile['homa_ir'] > global_median['homa_ir']
        high_testosterone = profile['testosterone'] > global_median['testosterone'] 
        high_bmi = profile['bmi'] > global_median['bmi']
        high_amh = profile['amh'] > global_median['amh']
        high_follicles = profile['total_follicles'] > global_median['total_follicles']
        irregular_cycle = profile['menstrual_regularity'] > 0.5
        
        # Logique d'interprétation basée sur la littérature médicale
        if high_ir and high_bmi and not high_testosterone:
            interpretations[cluster_id] = 'SOPK Métabolique (Insulinorésistance)'
        elif high_testosterone and not high_ir and not high_bmi:
            interpretations[cluster_id] = 'SOPK Androgénique (Hyperandrogénie)'
        elif high_ir and high_testosterone and high_bmi:
            interpretations[cluster_id] = 'SOPK Sévère (Combiné)'
        elif high_amh and high_follicles and irregular_cycle:
            interpretations[cluster_id] = 'SOPK Ovulatoire (AMH élevé)'
        elif not high_ir and not high_testosterone and irregular_cycle:
            interpretations[cluster_id] = 'SOPK Modéré (Cycle irrégulier)'
        else:
            interpretations[cluster_id] = f'SOPK Mixte (Cluster {cluster_id})'
    
    return interpretations

cluster_interpretation = interpret_pcos_subtypes(cluster_profiles)
print("\n=== INTERPRÉTATION DES SOUS-TYPES SOPK ===")
for cluster_id, interpretation in cluster_interpretation.items():
    cluster_size = (df_combined['cluster'] == cluster_id).sum()
    percentage = (cluster_size / len(df_combined)) * 100
    print(f"Cluster {cluster_id}: {interpretation}")
    print(f"  → {cluster_size} patientes ({percentage:.1f}%)")

df_combined['cluster_interpretation'] = df_combined['cluster'].map(cluster_interpretation)

# 11. RÉPARTITION DES SOUS-TYPES
plt.figure(figsize=(12, 6))
cluster_counts = df_combined['cluster_interpretation'].value_counts()
bars = plt.bar(cluster_counts.index, cluster_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Répartition des Sous-types de SOPK', fontsize=14, fontweight='bold')
plt.xlabel('Sous-type')
plt.ylabel('Nombre de patientes')
plt.xticks(rotation=45, ha='right')

# Ajouter les nombres sur les barres
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 12. CARACTÉRISTIQUES CLÉS PAR SOUS-TYPE
print("\n=== CARACTÉRISTIQUES CLÉS PAR SOUS-TYPE ===")
key_features = ['testosterone', 'homa_ir', 'bmi', 'amh', 'total_follicles']

# Calcul des médianes globales pour la comparaison
global_median = cluster_profiles.median()

for cluster_id in cluster_profiles.index:
    print(f"\nSous-type {cluster_id} - {cluster_interpretation[cluster_id]}:")
    profile = cluster_profiles.loc[cluster_id]
    for feature in key_features:
        value = profile[feature]
        median = global_median[feature]
        status = "↑ ÉLEVÉ" if value > median else "↓ FAIBLE" if value < median else "≈ MOYEN"
        print(f"  {feature}: {value:.2f} ({status})")

# 13. PROFILS DÉTAILLÉS DES SOUS-TYPES
print("\n=== PROFILS DÉTAILLÉS DES SOUS-TYPES SOPK ===")
for cluster_id in cluster_profiles.index:
    print(f"\n📊 SOUS-TYPE {cluster_id}: {cluster_interpretation[cluster_id]}")
    profile = cluster_profiles.loc[cluster_id]
    
    print("  CARACTÉRISTIQUES PRINCIPALES:")
    print(f"  • Testostérone: {profile['testosterone']:.1f} pg/mL")
    print(f"  • Insulinorésistance (HOMA-IR): {profile['homa_ir']:.2f}")
    print(f"  • IMC: {profile['bmi']:.1f} kg/m²")
    print(f"  • AMH: {profile['amh']:.1f} ng/mL")
    print(f"  • Follicles totaux: {profile['total_follicles']:.0f}")
    print(f"  • Règles irrégulières: {'Oui' if profile['menstrual_regularity'] > 0.5 else 'Non'}")

# 14. VALIDATION DES RÉSULTATS
print("\n=== VALIDATION ===")
final_silhouette = silhouette_score(X_scaled, cluster_labels)
print(f"Score de silhouette final: {final_silhouette:.4f}")

cluster_sizes = df_combined['cluster'].value_counts()
print("\nTaille des clusters:")
for cluster_id, size in cluster_sizes.items():
    percentage = (size / len(df_combined)) * 100
    print(f"Cluster {cluster_id}: {size} patientes ({percentage:.1f}%)")

# 15. SAUVEGARDE DES RÉSULTATS
#df_combined.to_csv('pcos_dataset_with_clusters.csv', index=False)
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca_model.pkl')

# Sauvegarde des profils de clusters
#cluster_profiles.to_csv('pcos_cluster_profiles.csv')

print("\n" + "="*60)
print("✅ SOUS-TYPAGE SOPK TERMINÉ AVEC SUCCÈS!")
print(f"✅ {optimal_k} sous-types identifiés")
print(f"✅ Score de silhouette: {final_silhouette:.4f}")
print(f"✅ {len(df_combined)} patientes analysées")
print("✅ Résultats sauvegardés")
print("="*60)

# 16. RAPPORT FINAL DES SOUS-TYPES
print("\n" + "🔬 RAPPORT FINAL DES SOUS-TYPES SOPK 🔬")
print("="*50)
for cluster_id in cluster_profiles.index:
    size = (df_combined['cluster'] == cluster_id).sum()
    percentage = (size / len(df_combined)) * 100
    interpretation = cluster_interpretation[cluster_id]
    
    print(f"\n🎯 Sous-type {cluster_id}: {interpretation}")
    print(f"   👥 Prévalence: {size} patientes ({percentage:.1f}%)")
    
    profile = cluster_profiles.loc[cluster_id]
    # Identifier les caractéristiques dominantes
    dominant_features = []
    if profile['testosterone'] > global_median['testosterone']:
        dominant_features.append("Hyperandrogénie")
    if profile['homa_ir'] > global_median['homa_ir']:
        dominant_features.append("Insulinorésistance")
    if profile['bmi'] > global_median['bmi']:
        dominant_features.append("Obésité")
    if profile['amh'] > global_median['amh']:
        dominant_features.append("AMH élevé")
    if profile['total_follicles'] > global_median['total_follicles']:
        dominant_features.append("Polykystique")
    
    print(f"   📈 Caractéristiques: {', '.join(dominant_features)}")
print("="*50)