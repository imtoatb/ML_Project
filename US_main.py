import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def clustering(df):
 
    # Select features for clustering
    features_clustering = ['testosterone', 'homa_ir', 'bmi', 'amh', 'total_follicles']
    available_features = [f for f in features_clustering if f in df.columns]
    X_cluster = df[available_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    


    
    k_range = range(2, 8)                                                                               # Test k from 2 to 7
    silhouette_scores = []  
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)                                                      # Goal > 0.8   
        silhouette_scores.append(score)
        print(f"k={k}, silhouette: {score:.4f}")

    print("\n")

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.show()

 




    pca = PCA(n_components=2)                                                           #5D -> 2D        
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if pcos else 'blue' for pcos in df['pcos']]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)                          #alpha = opacity (0 transparent, 1 solid)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA (Red=PCOS, Blue=Non-PCOS)')
    plt.show()






    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_k= kmeans.fit_predict(X_scaled)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_k, cmap='viridis', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'K-means Clustering (k=4)')
    plt.show()





    if 'pcos' in df.columns:
        plt.figure(figsize=(10, 6))
        cluster_distribution = pd.crosstab(labels_k, df['pcos'], normalize='index')
        cluster_distribution.plot(kind='bar', color=['blue', 'red'])
        plt.title('Diagnostic Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Proportion')
        plt.legend(['Non-PCOS', 'PCOS'])
        plt.show()

  



    best_silhouette = max(silhouette_scores)
    print(f"Best silhouette score: {best_silhouette:.3f} (goal > 0.8)")


df = pd.read_csv('final_merged_dataset.csv')
clustering(df)