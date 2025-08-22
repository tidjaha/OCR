import streamlit as st
import io
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from doctr.io import DocumentFile
from doctr.models import recognition

# Configuration de la page
st.set_page_config(page_title="OCR App", layout="wide")

# Titre et introduction
st.title("Bienvenue dans mon projet OCR !")
st.write("Ce travail s'inscrit dans le cadre de ma formation en deep learning.")

# Chargement des modèles
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("best.pt")
        ocr_model = recognition.crnn_vgg16_bn(pretrained=True).eval()
        return yolo_model, ocr_model, True
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        return None, None, False

yolo_model, ocr_model, models_loaded = load_models()

# Interface principale
def main():
    # Étape 1: Upload de l'image
    uploaded_file = st.file_uploader("Choisis une image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image importée", use_container_width=True)
        
        # Étape 2: Confirmation de l'image
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Oui, utiliser cette image"):
                process_image(image, yolo_model, ocr_model)
        with col2:
            if st.button("Non, choisir une autre image"):
                st.rerun()

# Fonction de traitement d'image
def process_image(image, yolo_model, ocr_model):
    img_array = np.array(image)
    
    # Détection avec YOLO
    with st.spinner("Détection des caractères en cours..."):
        results = yolo_model.predict(img_array)
        im_annotated = results[0].plot()
        st.image(im_annotated, caption="Détection YOLO", use_container_width=True)
    
    # Extraire les boxes
    boxes = []
    for i, box in enumerate(results[0].boxes):
        conf = float(box.conf[0])
        if conf < 0.33:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        boxes.append([x1, y1, x2, y2, conf])
    
    if not boxes:
        st.warning("⚠️ Aucune box détectée avec assez de confiance.")
        return
    
    # NMS simple
    def iou(box1, box2):
        x1, y1, x2, y2 = box1[:4]
        X1, Y1, X2, Y2 = box2[:4]
        inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
        inter_x2, inter_y2 = min(x2, X2), min(y2, Y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (X2 - X1) * (Y2 - Y1)
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0
    
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    nms_boxes = []
    while boxes:
        best = boxes.pop(0)
        nms_boxes.append(best)
        boxes = [b for b in boxes if iou(best, b) < 0.4]
    
    # Trier par ligne
    heights = [y2 - y1 for x1, y1, x2, y2, _ in nms_boxes]
    avg_h = int(np.mean(heights)) if heights else 1
    boxes_sorted = sorted(nms_boxes, key=lambda b: (b[1] // avg_h, b[0]))
    
    # OCR avec DocTR
    recognized_text = []
    st.subheader("Reconnaissance des caractères")
    
    for idx, (x1, y1, x2, y2, conf) in enumerate(boxes_sorted, start=1):
        crop = img_array[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop)
        
        buffer = io.BytesIO()
        crop_pil.save(buffer, format="JPEG")
        buffer.seek(0)
        
        doc = DocumentFile.from_images(buffer.getvalue())
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(crop_pil, caption=f"Crop {idx}", use_container_width=True)
        
        img = doc[0].astype("float32") / 255.0
        
        h, w, _ = img.shape
        if h != 32:
            new_w = int(w * 32 / h)
            resized = cv2.resize(img, (new_w, 32))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            out = ocr_model(tensor)
        
        if "preds" in out:
            recognized_char = out["preds"][0][0]
        else:
            recognized_char = "?"
        
        recognized_text.append(recognized_char)
        with col2:
            st.write(f"**Caractère {idx}:** {recognized_char}")
            st.write(f"Confiance: {conf:.2f}")
    
    # Afficher texte final
    st.subheader("📝 Texte Reconnu")
    final_text = " ".join(recognized_text)
    st.success(final_text)
    
    st.download_button(
        "📥 Télécharger le texte",
        data=final_text,
        file_name="ocr_result.txt",
        mime="text/plain",
    )
    
    if st.button("🔄 Analyser une nouvelle image"):
        st.rerun()

# Lancement de l'application
if __name__ == "__main__":
    if models_loaded:
        main()
    else:
        st.error("Les modèles ne sont pas chargés correctement. Impossible de continuer.")
