import os
from flask import Flask, render_template, request, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import logging
import tensorflow as tf
from datetime import datetime  # Importation de datetime pour l'heure et la date

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Constantes
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'  # Dossier de téléchargement des images
TARGET_SIZE = (150, 150)  # Taille attendue par le modèle
PASSWORD = "FadmaTohmi"  # Mot de passe

# Création du dossier d'uploads s'il n'existe pas
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = None

def load_model():
    """Charger le modèle TensorFlow"""
    global model
    try:
        if os.path.exists('Abdessalamsib.h5'):
            model = tf.keras.models.load_model('Abdessalamsib.h5')
            logger.info("Modèle chargé avec succès")
            return True
        else:
            logger.error("Fichier modèle non trouvé: Abdessalamsib.h5")
            return False
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Prétraitement de l'image pour le modèle"""
    try:
        img = Image.open(image_path)
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img)

        # Convertir en RGB si l'image est en niveaux de gris
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]  # Supprimer le canal alpha

        img_array = img_array / 255.0  # Normalisation
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
        return img_array
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image: {str(e)}")
        return None

def predict_image(image_array):
    """Faire une prédiction avec le modèle"""
    if model is None:
        if not load_model():
            return None, "Erreur: Le modèle n'est pas disponible."

    try:
        prediction = model.predict(image_array)
        probability = float(prediction[0][0])
        result = "Aucune tumeur détectée" if probability >0.5 else "Tumeur détectée"
        return 100-probability * 100, result  # Retourner le pourcentage et le résultat
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        return None, "Erreur lors de l'analyse de l'image."

def enregistrer_utilisateur(nom, prenom, mot_de_passe):
    """Enregistrer les informations de l'utilisateur dans un fichier texte avec date et heure"""
    try:
        date_heure = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Récupération de la date et l'heure actuelle
        with open("utilisateurs.txt", "a") as file:
            file.write(f"{nom} {prenom}, Mot de passe: {mot_de_passe}, Date et Heure: {date_heure}\n")
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement de l'utilisateur: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'authentifie' not in session:
        if request.method == 'POST':
            nom = request.form.get('nom')
            prenom = request.form.get('prenom')
            mot_de_passe = request.form.get('mot_de_passe')

            # Vérification du mot de passe
            if mot_de_passe == PASSWORD:
                session['authentifie'] = True
                enregistrer_utilisateur(nom, prenom, mot_de_passe)
                flash("Bienvenue, vous êtes authentifié!", 'success')
                return redirect(url_for('analyse_image'))  # Redirige vers la page d'analyse après l'authentification
            else:
                flash("Mot de passe incorrect. Essayez encore.", 'error')

        return render_template('login.html')

    # Si l'utilisateur est authentifié, afficher la page d'accueil
    return redirect(url_for('analyse_image'))  # Redirige vers la page d'analyse si l'utilisateur est déjà authentifié

@app.route('/analyse_image', methods=['GET', 'POST'])
def analyse_image():
    if 'authentifie' not in session:
        return redirect(url_for('home'))  # Si l'utilisateur n'est pas authentifié, rediriger vers la page de connexion

    if request.method == 'POST':
        if 'mri_image' not in request.files:
            flash('Aucun fichier sélectionné', 'error')
            return render_template('home.html')

        file = request.files['mri_image']
        if file.filename == '':
            flash('Aucun fichier sélectionné', 'error')
            return render_template('home.html')

        if not allowed_file(file.filename):
            flash('Type de fichier invalide. Veuillez télécharger une image PNG ou JPG.', 'error')
            return render_template('home.html')

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Prétraitement et prédiction
            processed_image = preprocess_image(filepath)
            if processed_image is not None:
                probability, prediction = predict_image(processed_image)
                session['prediction'] = prediction
                session['probability'] = f"{probability:.2f}"  # Format the probability to two decimal places
                session['image_path'] = url_for('static', filename='uploads/' + filename)  # Generate the URL for the image
                flash('Analyse complétée!', 'success')
            else:
                flash('Erreur lors du traitement de l\'image', 'error')
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier: {str(e)}")
            flash('Erreur lors du traitement de l\'image', 'error')
    
    return render_template('home.html', prediction=session.get('prediction'),
                           probability=session.get('probability'),
                           image_path=session.get('image_path'))

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
