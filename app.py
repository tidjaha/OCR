import streamlit as st
import io
import cv2
import numpy as np
from PIL import Image
import os, glob, re, torch
from ultralytics import YOLO
from doctr.io import DocumentFile
from doctr.models import recognition


#chargement des modeles :
yolo_model = YOLO("best.pt")  # ton modèle YOLO entraîné
ocr_model = recognition.crnn_vgg16_bn(pretrained=True).eval()

#petite presentation
st.title("Bienvenue dans mon projet OCR ! Ce travail s’inscrit dans le cadre de ma formation en deep learning, qui est encore en cours et devrait se terminer dans environ un mois.")




# Zone vide pour tout le bloc upload + confirmation
placeholder = st.empty()

# État de confirmation
if "image_confirmee" not in st.session_state:
    st.session_state.image_confirmee = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Fonction reset

def reset_app():
    st.session_state.uploaded_file = None
    st.session_state.image_confirmee = False
    placeholder.empty()

with placeholder.container():
    uploaded_file = st.file_uploader(
        "Choisis une image à analyser",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        image = Image.open(uploaded_file)
        st.image(image, caption="Image importée", use_column_width=True)    
        st.write("C'est bien votre image ?", ("Oui", "Non"))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Oui"):
                st.session_state.image_confirmee = True
                st.success("Vous avez confirmé. On continue !")
                img_array = np.array(image)
            
                # -----------------------------
                # Étape 1 - Détection avec YOLO
                # -----------------------------
                results = yolo_model.predict(img_array)
                im_annotated = results[0].plot()
                st.image(im_annotated, caption="Détection YOLO", use_column_width=True)
            
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
                else:
                    # Supprimer chevauchements (NMS simple)
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
            
                    # Trier par ligne (y puis x)
                    heights = [y2 - y1 for x1, y1, x2, y2, _ in nms_boxes]
                    avg_h = int(np.mean(heights))
                    boxes_sorted = sorted(nms_boxes, key=lambda b: (b[1] // avg_h, b[0]))
            
                    # -----------------------------
                    # Étape 2 - OCR avec DocTR
                    # -----------------------------
                    recognized_text = []
                    for idx, (x1, y1, x2, y2, conf) in enumerate(boxes_sorted, start=1):
                        # ton crop numpy
                        crop = img_array[y1:y2, x1:x2]
                        
                        # conversion en image PIL
                        crop_pil = Image.fromarray(crop)
                        
                        # sauvegarde en mémoire (buffer) au format JPEG
                        buffer = io.BytesIO()
                        crop_pil.save(buffer, format="JPEG")
                        buffer.seek(0)  # remettre le curseur au début du buffer
                        
                        # utiliser dans doctr
                        doc = DocumentFile.from_images(buffer.getvalue())
                        
                        # afficher dans Streamlit
                        st.image(crop_pil, caption="Aperçu du crop", use_container_width=True)
            
                        
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
            
                        # Récupération du caractère prédit
                        if "preds" in out:
                            recognized_char = out["preds"][0][0]
                        else:
                            recognized_char = "?"
            
                        recognized_text.append(recognized_char)
                        st.image(crop, caption=f"Caractère {idx}: {recognized_char}", width=80)
            
                    # -----------------------------
                    # Étape 3 - Afficher texte final
                    # -----------------------------
                    st.subheader("📝 Texte Reconnu")
                    final_text = " ".join(recognized_text)
                    st.write(final_text)
            
                    # Option téléchargement
                    st.download_button(
                        "📥 Télécharger le texte",
                        data=final_text,
                        file_name="ocr_result.txt",
                        mime="text/plain",
                    )
        with col2:
            if st.button("Non"):
                reset_app()
                



