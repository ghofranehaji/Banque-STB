import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import joblib

def main():
    st.title("🔍 Estimation du segment carte de crédit")
    st.markdown("Cette application estime le segment d'un client selon un clustering K-means déjà réalisé sur les données historiques.")

    # === 1. Chargement des données ===
    try:
        df = pd.read_csv('clustering_carte_credit_kmeans.csv', sep=";", encoding='latin-1')
    except FileNotFoundError:
        st.error("❌ Fichier 'clustering_carte_credit_kmeans.csv' non trouvé. Vérifiez le chemin du fichier.")
        return

    # === 2. Colonnes utilisées pour la prédiction ===
    features = ['Nombres_cheques_par_client', 'salaire', 'Frequence_Paiements', 'Total_des_cheques']
    required_cols = features + ['segment_carte_credit']
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Certaines colonnes nécessaires sont manquantes dans le fichier.")
        return

    X = df[features]

    # === 3. Standardisation ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === 4. DataFrame standardisé avec segment ===
    df_scaled = pd.DataFrame(X_scaled, columns=features)
    df_scaled['segment'] = df['segment_carte_credit']

    # === 5. Calcul des centroïdes ===
    centroids = df_scaled.groupby('segment').mean().values
    segments_labels = df_scaled.groupby('segment').mean().index.tolist()

    # === Dictionnaire des noms des clusters ===
    cluster_names = {
        1: 'Clients Premium',
        2: 'Utilisateurs Occasionnels',
        3: 'Clients Fidèles',
        4: 'Profils à Consolider',
        50: 'Clients à Haut Potentiel'
    }

    # === 6. Limites pour détection anomalies ===
    limits = {}
    for feature in features:
        q1 = df[feature].quantile(0)
        q3 = df[feature].quantile(1)
        iqr = q3 - q1
        limits[feature] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # === 7. Interface utilisateur ===
    st.subheader("🧾 Entrer les informations du client")
    nombres_cheques = st.number_input("Nombre de chèques par client", min_value=0)
    salaire = st.number_input("Salaire", min_value=0)
    frequence = st.number_input("Fréquence des paiements", min_value=0)
    total_cheques = st.number_input("Total des chèques", min_value=0)

    if st.button("📊 Prédire le segment"):
        new_data = {
            'Nombres_cheques_par_client': nombres_cheques,
            'salaire': salaire,
            'Frequence_Paiements': frequence,
            'Total_des_cheques': total_cheques
        }

        anomalies = [f"{k} ({v})" for k, v in new_data.items() if not (limits[k][0] <= v <= limits[k][1])]

        if anomalies:
            st.warning(f"❌ Valeurs anormales détectées pour : {', '.join(anomalies)}. Veuillez corriger les saisies.")
        else:
            new_client = np.array([[nombres_cheques, salaire, frequence, total_cheques]])
            new_client_scaled = scaler.transform(new_client)
            distances = cdist(new_client_scaled, centroids)
            closest = distances.argmin()
            predicted_segment = segments_labels[closest]

            predicted_label = cluster_names.get(predicted_segment, f"Cluster {predicted_segment}")
            st.success(f"✅ Segment estimé du client : **{predicted_label}**")

    # === 8. Option sauvegarde ===
    if st.checkbox("💾 Sauvegarder le scaler et les centroïdes"):
        joblib.dump(scaler, 'scaler.pkl')
        np.save('centroids.npy', centroids)
        np.save('segments_labels.npy', np.array(segments_labels))
        st.info("✅ Sauvegarde effectuée avec succès.")

if __name__ == "__main__":
    main()
