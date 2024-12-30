import os
import uuid
import zipfile

from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

import torch
import numpy as np
from PIL import Image
from utils.config import Config, ClassMapper
from utils.augmentation import get_augmentation_pipeline
from models.swin_transformer import get_swin_model

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    SamAutomaticMaskGenerator = None  

# ======================================================================
#                          CONFIG FLASK
# ======================================================================
app = Flask(__name__)
app.secret_key = "votre_cle_secret_flask"

app.config["UPLOAD_FOLDER"] = os.path.join("src/static", "images")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "zip"}
class_mapper = ClassMapper()

# ======================================================================
#                      FONCTIONS UTILITAIRES
# ======================================================================
def allowed_file(filename):
    """
    Vérifie que l'extension du fichier est autorisée (images ou .zip).
    """
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS

def extract_zip_to_folder(zip_path, destination_folder):
    """
    Extrait un fichier .zip dans le dossier destination_folder.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination_folder)

def load_swin_model(model_path):
    """
    Charge le modèle Swin Transformer avec ses poids pré-entraînés.
    """
    model_swin, _, name = get_swin_model(Config.NUM_CLASSES)
    model_swin.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model_swin.eval()
    return model_swin, name

def predict_class(model, image_path):
    """
    Prédit la classe et la probabilité pour une image via le modèle Swin.
    Retourne (classe_predite, probabilité).
    """
    img_pil = Image.open(image_path).convert("RGB")
    pipeline_augmentation = get_augmentation_pipeline(train=False)
    img_tensor = pipeline_augmentation(img_pil).unsqueeze(0)  # shape [1, C, H, W]

    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=False)
        logits = outputs.logits  # si votre modèle renvoie logits via la clé `logits`
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_idx = torch.argmax(probs, dim=-1).item()
        predicted_prob = torch.max(probs, dim=-1).values.item()
    
    return predicted_idx, predicted_prob

def visualize_attention(model_path, image_path, combine_all=True):
    """
    Génère un overlay (global attention).
    Retourne le chemin relatif vers l'image d'attention.
    """
    # Import de vos fonctions (eyes_of_models)
    from eyes_of_models import (
        visualize_attention_maps,
        plot_global_attention
    )

    global_att, all_attention_maps, outputs = visualize_attention_maps(
        model_path=model_path,
        image_path=image_path,
        combine_all=combine_all,
        vizu=False
    )

    # On sauvegarde l’overlay
    overlay_name = "attention_overlay.png"
    overlay_path = os.path.join(app.config["UPLOAD_FOLDER"], overlay_name)

    image_pil = Image.open(image_path).convert("RGB")
    plot_global_attention(
        image_pil, 
        global_att,
        save_path=overlay_path
    )
    return os.path.join("images", overlay_name)

def visualize_layerwise_attention(model_path, image_path):
    """
    Génère une visualisation layer-wise (toutes les couches).
    Retourne le chemin relatif d'un montage PNG pour affichage.
    """
    from eyes_of_models import (
        visualize_attention_maps,
        plot_layerwise_attention
    )

    all_attention_maps, outputs = visualize_attention_maps(
        model_path=model_path,
        image_path=image_path,
        combine_all=False,
        vizu=False
    )

    layerwise_dir = os.path.join("static", "attention", "layers")
    os.makedirs(layerwise_dir, exist_ok=True)
    output_path = os.path.join(layerwise_dir, "layers_attention.png")

    image_pil = Image.open(image_path).convert("RGB")
    plot_layerwise_attention(image_pil, all_attention_maps, save_path=output_path)

    # Retourne le chemin relatif depuis "static/"
    return output_path.split("static/")[-1]

def segment_with_sam(image_path):
    """
    Segmente l'image avec SAM (Segment Anything).
    Retourne le chemin relatif vers un overlay (rouge) du premier masque.
    """
    if SamAutomaticMaskGenerator is None:
        print("SAM n'est pas installé ou importé. Retourne None.")
        return None

    # Charger l'image
    image_pil = Image.open(image_path).convert("RGB")
    image_array = np.array(image_pil)

    # Charger le modèle SAM
    sam_checkpoint = os.path.join("models", "sam_vit_h_4b8939.pth")
    sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam_model.eval()

    mask_generator = SamAutomaticMaskGenerator(sam_model)
    masks = mask_generator.generate(image_array)
    if len(masks) == 0:
        return None

    # Utiliser le 1er masque pour la démo
    mask0 = masks[0]["segmentation"].astype(np.uint8) * 255  
    mask0_pil = Image.fromarray(mask0).convert("RGBA")

    overlay = Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    overlay_data = overlay.load()
    mask_data = mask0_pil.load()

    for y in range(image_pil.height):
        for x in range(image_pil.width):
            if mask_data[x, y][0] > 0:
                # On colore en rouge avec alpha 100
                overlay_data[x, y] = (255, 0, 0, 100)

    result = Image.alpha_composite(image_pil.convert("RGBA"), overlay)

    seg_filename = "segment_sam.png"
    seg_path = os.path.join(app.config["UPLOAD_FOLDER"], seg_filename)
    result.save(seg_path)

    return os.path.join("images", seg_filename)

# ======================================================================
#                           ROUTES FLASK
# ======================================================================

@app.route("/")
def index():
    """
    Page d'accueil - On liste les images déjà uploadées.
    """
    file_list = os.listdir(app.config["UPLOAD_FOLDER"])
    # On ne garde que les images png/jpg/jpeg
    images_list = [
        f for f in file_list 
        if "." in f and f.rsplit(".", 1)[1].lower() in {"png","jpg","jpeg"}
    ]
    return render_template("index.html", images=images_list)

@app.route("/upload", methods=["POST"])
def upload():
    """
    Permet de charger :
    - plusieurs images
    - ou un ZIP (contenant un dossier d'images)
    """
    files = request.files.getlist("files[]")
    if not files:
        return redirect(url_for("index"))

    for f in files:
        if f.filename == "":
            continue
        if allowed_file(f.filename):
            # Vérifie si c'est un zip
            ext = f.filename.rsplit(".", 1)[1].lower()
            if ext == "zip":
                zip_unique_name = str(uuid.uuid4()) + "_" + secure_filename(f.filename)
                zip_path = os.path.join(app.config["UPLOAD_FOLDER"], zip_unique_name)
                f.save(zip_path)
                # Décompresser
                extract_zip_to_folder(zip_path, app.config["UPLOAD_FOLDER"])
                # On peut ensuite supprimer le zip
                os.remove(zip_path)
            else:
                # Sinon, c'est une image
                filename = secure_filename(f.filename)
                unique_name = str(uuid.uuid4()) + "_" + filename
                f.save(os.path.join(app.config["UPLOAD_FOLDER"], unique_name))

    return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    """
    Retourne la classe prédite et la probabilité pour l'image sélectionnée.
    """
    data = request.get_json()
    image_name = data["image_name"]
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_name)

    model_swin, _ = load_swin_model(os.path.join("models", "Swin.pth"))
    pred_class, prob = predict_class(model_swin, image_path)

    class_text = class_mapper.class2name.get(pred_class, f"Unknown class {pred_class}")

    return jsonify({
        "predicted_class": class_text,
        "probability": round(prob, 4)
    })

@app.route("/attention", methods=["POST"])
def attention():
    """
    Retourne l'URL de l'image d'attention (overlay).
    """
    data = request.get_json()
    image_name = data["image_name"]
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_name)

    attention_rel_path = visualize_attention(
        model_path=os.path.join("models", "Swin.pth"),
        image_path=image_path,
        combine_all=True
    )

    return jsonify({
        "attention_image_url": url_for("static", filename=attention_rel_path)
    })

@app.route("/segment", methods=["POST"])
def segment():
    """
    Retourne l'URL de l'image segmentée via SAM.
    """
    data = request.get_json()
    image_name = data["image_name"]
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_name)

    seg_rel_path = segment_with_sam(image_path)
    if seg_rel_path is None:
        return jsonify({"segmented_image_url": ""}), 400

    return jsonify({
        "segmented_image_url": url_for("static", filename=seg_rel_path)
    })

@app.route("/advanced_attention/<img_name>")
def advanced_attention(img_name):
    """
    Page qui affiche un montage layer-wise de l'attention.
    """
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], img_name)
    layerwise_rel_path = visualize_layerwise_attention(
        model_path=os.path.join("models", "Swin.pth"),
        image_path=image_path
    )
    # ex: layerwise_rel_path = "attention/layers/layers_attention.png"
    return render_template("advanced_attention.html",
                           original_image=img_name,
                           attention_layers_image=url_for("static", filename=layerwise_rel_path))

if __name__ == "__main__":
    app.run(debug=True)