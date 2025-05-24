import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import joblib

def main():
    st.title("🔍 Estimation du segment carte de crédit")
    st.markdown("Cette application estime le segment d'un client selon un clustering hiérarchique déjà réalisé sur les données historiques.")

    # === 1. Chargement des données ===
    try:
        df = pd.read_csv('clustering_carte_credit.csv', sep=";", encoding='latin-1')
    except FileNotFoundError:
        st.error("❌ Fichier 'clustering_carte_credit.csv' non trouvé. Vérifiez le chemin du fichier.")
        return

    # === 2. Colonnes utilisées pour la prédiction ===
    features = ['Age', 'salaire', 'Frequence_Paiements', 'Total_des_cheques']
    if not all(f in df.columns for f in features + ['segment_carte_credit']):
        st.error("❌ Certaines colonnes nécessaires sont manquantes dans le fichier.")
        return

    X = df[features]

    # === 3. Standardisation ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === 4. Création du DataFrame standardisé avec segments ===
    df_scaled = pd.DataFrame(X_scaled, columns=features)
    df_scaled['segment'] = df['segment_carte_credit']

    # === 5. Calcul des centroïdes de chaque segment (approche manuelle) ===
    centroids = df_scaled.groupby('segment').mean().values
    segments_labels = df_scaled.groupby('segment').mean().index.tolist()

    # === 6. Détection des valeurs aberrantes pour chaque variable ===
    limits = {}
    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        min_valid = q1 - 1.5 * iqr
        max_valid = q3 + 1.5 * iqr
        limits[feature] = (min_valid, max_valid)

    # === 7. Interface utilisateur ===
    st.subheader("🧾 Entrer les informations du client")
    age = st.number_input("Âge", min_value=18, max_value=100)
    salaire = st.number_input("Salaire", min_value=450, max_value=10000)
    frequence = st.number_input("Fréquence des paiements", min_value=1, max_value=1000)
    total_cheques = st.number_input("Total des chèques", min_value=1, max_value=100000)

    if st.button("📊 Prédire le segment"):
        new_data = {'Age': age, 'salaire': salaire, 'Frequence_Paiements': frequence, 'Total_des_cheques': total_cheques}
        anomalies = [f"{k} ({v})" for k, v in new_data.items() if not (limits[k][0] <= v <= limits[k][1])]

        if anomalies:
            st.warning(f"❌ Valeurs anormales détectées pour : {', '.join(anomalies)}. Veuillez corriger les saisies.")
        else:
            new_client = np.array([[age, salaire, frequence, total_cheques]])
            new_client_scaled = scaler.transform(new_client)

            # Distance du client à chaque centroïde
            distances = cdist(new_client_scaled, centroids)
            closest = distances.argmin()
            predicted_segment = segments_labels[closest]
            st.success(f"✅ Segment estimé du client : **{predicted_segment}**")

    # === 8. Option de sauvegarde ===
    if st.checkbox("💾 Sauvegarder le scaler et les centroïdes"):
        joblib.dump(scaler, 'scaler.pkl')
        np.save('centroids.npy', centroids)
        np.save('segments_labels.npy', np.array(segments_labels))
        st.info("✅ Sauvegarde effectuée avec succès.")

# ✅ Point d'entrée
if __name__ == "__main__":
    main()
