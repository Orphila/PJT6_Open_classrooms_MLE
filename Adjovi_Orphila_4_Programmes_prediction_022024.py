import streamlit as st
import mlflow
from PIL import Image
import numpy as np

# Charger le modèle MLflow
best_model_uri = "file:///Users/orphila_adjovi/shelf_capacity_prediction/DEV/mlruns/0/c06fd0c661c348dbbd866b69c7d25cc9/artifacts/mdl "
model = mlflow.load_model(best_model_uri)

# Prétraitement + prédiction
def predict_dog_breed(image):
    # Redimensionnement
    resized_image = image.resize((224, 224))

    image_array = np.array(resized_image) / 255.0
    # Ajout d'une dimension supplémentaire pour correspondre au format d'entrée du modèle (224, 224, 3)
    input_image = np.expand_dims(image_array, axis=0)

    # Prédiction sur l'image prétraitée
    prediction = model.predict(input_image)

    return prediction

# Streamlit
st.title("Prédiction de race de chien")

uploaded_file = st.file_uploader("Téléchargez une image de chien", type=["jpg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # pred
    prediction = predict_dog_breed(image)

    # Résultat de la prédiction
    st.subheader("Résultat de la prédiction :")
    st.write(prediction)
