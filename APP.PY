import os
import glob
import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import zipfile
# Auto-descomprimir las imágenes si existe el archivo zip
if not os.path.exists("IR_Lab") and os.path.exists("IR LAB.zip"):
    with zipfile.ZipFile("IR LAB.zip", "r") as zip_ref:
        zip_ref.extractall(".")
from PIL import Image
from scipy.signal import find_peaks, savgol_filter

# ---------------- CONFIGURACIÓN DE PÁGINA Y ESTILO ----------------
st.set_page_config(
    page_title="SpectroAnalyzer PRO",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Poppins:wght@500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
    }
    
    .hero-box {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #e1e8ed;
        margin-bottom: 25px;
    }
    
    .hero-title {
        background: linear-gradient(135deg, #E94A5C 0%, #4A90E2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }
    
    .success-card {
        background-color: #fdf3f4;
        border: 1px solid #fad5d9;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
    }
    
    .info-card {
        background-color: #eef2f5;
        border-left: 5px solid #4A90E2;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIGURACIÓN DE RUTAS Y BASE DE DATOS ----------------
BASE_DIR = "IR_Lab"
os.makedirs(BASE_DIR, exist_ok=True)
EXTENSIONES = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif"]

GRUPOS = {
    "Alcanos": {"folder": "alcanos", "desc": "Compuestos saturados. Presentan vibraciones de tensión C-H sp³ por debajo de 3000 cm⁻¹ y flexión C-H alrededor de 1450 y 1375 cm⁻¹."},
    "Alquenos": {"folder": "alquenos", "desc": "Hidrocarburos insaturados. Muestran señales de =C-H sp² por encima de 3000 cm⁻¹ y tensión C=C (generalmente débil) entre 1620 y 1680 cm⁻¹."},
    "Alquinos": {"folder": "alquinos", "desc": "Presentan un enlace triple. Tensión ≡C-H aguda en ~3300 cm⁻¹ (si es terminal) y tensión C≡C entre 2100 y 2260 cm⁻¹."},
    "Alcoholes": {"folder": "alcoholes", "desc": "Caracterizados por una banda O-H muy ancha y fuerte (por puentes de hidrógeno) entre 3200 y 3600 cm⁻¹, además de tensión C-O cerca de 1050-1150 cm⁻¹."},
    "Aldehídos": {"folder": "aldehidos", "desc": "Poseen un carbonilo terminal. Tensión C=O intensa (~1725 cm⁻¹) y un doblete característico C-H aldehídico por resonancia de Fermi en 2720 y 2820 cm⁻¹."},
    "Cetonas": {"folder": "cetonas", "desc": "Carbonilo interno. Banda C=O muy fuerte entre 1705 y 1725 cm⁻¹. Ausencia de bandas O-H o C-H aldehídicas."},
    "Ácidos Carboxílicos": {"folder": "acidos_carboxilicos", "desc": "Banda O-H extremadamente ancha y difusa (2500-3300 cm⁻¹) que suele solaparse con los C-H, sumada a una banda C=O intensa en ~1710 cm⁻¹."},
    "Aminas": {"folder": "aminas", "desc": "Tensión N-H en 3300-3500 cm⁻¹. Las aminas primarias muestran dos picos (simétrica y asimétrica), las secundarias un solo pico, y las terciarias ninguno."},
    "Haluros de ácido": {"folder": "haluros_de_acido", "desc": "Derivados de ácido donde el halógeno retira densidad electrónica. El pico C=O se desplaza a frecuencias muy altas (1780-1815 cm⁻¹)."},
    "Anhídridos": {"folder": "anhidridos", "desc": "Contienen dos grupos carbonilo acoplados. Muestran dos bandas de tensión C=O separadas por unos 60 cm⁻¹ (ej. 1760 y 1820 cm⁻¹)."}
}

# ---------------- FUNCIONES DE ANÁLISIS ----------------
def identificar_compuesto(picos_detectados):
    if not picos_detectados:
        return "<div style='color: #856404; background-color: #fff3cd; padding: 15px; border-radius: 8px;'>No hay picos suficientes para analizar. Ajusta la sensibilidad o los recortes.</div>"

    wns = [p['wn'] for p in picos_detectados]

    picos_carbonilo = [w for w in wns if 1670 <= w <= 1850]
    tiene_carbonilo = len(picos_carbonilo) > 0
    carbonilo_alto = any(1775 <= w <= 1850 for w in picos_carbonilo)
    doblete_anhidrido = len([w for w in picos_carbonilo if 1740 <= w <= 1850]) >= 2

    tiene_oh_alcohol = any(3200 <= w <= 3650 for w in wns)
    tiene_oh_acido = any(2500 <= w <= 2680 for w in wns) 
    tiene_amina = any(3300 <= w <= 3500 for w in wns)

    tiene_ch_aldehido = any(2680 <= w <= 2850 for w in wns)
    tiene_ch_sp2 = any(3000 <= w <= 3150 for w in wns)
    tiene_ch_sp3 = any(2850 <= w <= 3000 for w in wns)
    tiene_alquino_terminal = any(3250 <= w <= 3350 for w in wns)

    tiene_triple_enlace = any(2100 <= w <= 2260 for w in wns)
    tiene_doble_enlace = any(1620 <= w <= 1680 for w in wns)

    grupos_encontrados = []

    if tiene_carbonilo:
        if doblete_anhidrido:
            grupos_encontrados.append("<b>Anhídrido</b><br>Doblete de C=O a altas frecuencias (~1750 y 1820 cm⁻¹).")
        elif carbonilo_alto:
            grupos_encontrados.append("<b>Haluro de Ácido</b><br>C=O desplazado a frecuencias muy altas (>1775 cm⁻¹).")
        elif tiene_ch_aldehido:
            grupos_encontrados.append("<b>Aldehído</b><br>C=O intenso + señales C-H aldehídicas detectadas en 2700-2850 cm⁻¹.")
        elif tiene_oh_acido:
            grupos_encontrados.append("<b>Ácido Carboxílico</b><br>C=O intenso + absorción muy ancha O-H detectada en la base.")
        else:
            grupos_encontrados.append("<b>Cetona / Éster</b><br>Banda C=O fuerte detectada. Ausencia de O-H o C-H aldehídico.")
    else:
        if tiene_oh_alcohol:
            grupos_encontrados.append("<b>Alcohol</b><br>Banda O-H fuerte detectada entre 3200-3600 cm⁻¹ sin C=O.")
        elif tiene_amina:
            grupos_encontrados.append("<b>Amina</b><br>Banda N-H detectada en la región 3300-3500 cm⁻¹.")

    if tiene_triple_enlace or tiene_alquino_terminal:
        grupos_encontrados.append("<b>Alquino / Nitrilo</b><br>Posible banda en zona de triples enlaces.")
    if tiene_doble_enlace and tiene_ch_sp2:
        grupos_encontrados.append("<b>Alqueno / Aromático</b><br>Señales conjuntas de estiramiento C=C y =C-H sp².")
    if tiene_ch_sp3 and len(grupos_encontrados) == 0:
        grupos_encontrados.append("<b>Alcano</b><br>Solo estiramientos C-H alifáticos sp³ sin otros grupos funcionales reactivos.")

    if grupos_encontrados:
        detalles = "<div class='success-card'><h3 style='color: #E94A5C; margin-top:0;'>🎯 Veredicto Estructural</h3><ul>"
        for g in grupos_encontrados:
            detalles += f"<li>{g}</li>"
        detalles += "</ul></div>"
    else:
        detalles = "<div style='color: #856404; background-color: #fff3cd; padding: 15px; border-radius: 8px;'>Espectro no concluyente. Las señales extraídas no concuerdan strictly con los 10 grupos. Verifica los recortes.</div>"

    return detalles

def analizar_imagen_ir(imagen, wn_izq, wn_der, sensibilidad, x_inicio, x_fin, y_inicio, y_fin):
    img = np.array(imagen.convert("RGB"))
    alto, ancho = img.shape[:2]

    x0, x1 = max(0, int(ancho * x_inicio / 100)), min(ancho, int(ancho * x_fin / 100))
    y0, y1 = max(0, int(alto * y_inicio / 100)), min(alto, int(alto * y_fin / 100))
    if x0 >= x1 or y0 >= y1:
        st.error("El área de recorte es inválida. Revisa los deslizadores.")
        return None, None, None

    recorte = img[y0:y1, x0:x1]
    
    hsv = cv2.cvtColor(recorte, cv2.COLOR_RGB2HSV)
    gris = cv2.cvtColor(recorte, cv2.COLOR_RGB2GRAY)

    mascara_negra = ((gris < 160) & (hsv[:, :, 1] < 40)).astype(np.uint8) * 255
    h, w = mascara_negra.shape
    ys = np.full(w, np.nan)

    for x in range(w):
        filas = np.where(mascara_negra[:, x])[0]
        if len(filas):
            ys[x] = np.median(filas)

    valido = np.isfinite(ys)
    if float(np.mean(valido)) < 0.05:
        st.error("No pude detectar la curva continua. Ajusta los recortes para excluir texto.")
        return None, None, None

    columnas = np.arange(w)
    ys = np.interp(columnas, columnas[valido], ys[valido])
    ys_suave = savgol_filter(ys, min(21, w if w % 2 else w - 1), 2) if w >= 21 else ys

    punto_cambio = int(w * 0.4)
    wn_izq_mitad = np.linspace(float(wn_izq), 2000, punto_cambio)
    wn_der_mitad = np.linspace(2000, float(wn_der), w - punto_cambio)
    wn = np.concatenate((wn_izq_mitad, wn_der_mitad))

    picos, prop = find_peaks(ys_suave, prominence=float(sensibilidad), distance=max(4, w // 120))
    resultados = [{"wn": float(wn[p]), "prom": float(prop["prominences"][i])} for i, p in enumerate(picos)]
    resultados.sort(key=lambda r: r["wn"], reverse=True)

    curva_grafico = -ys_suave

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    ax.plot(wn, curva_grafico, color='#4A90E2', linewidth=1.5, label="Espectro Procesado")
    
    if len(picos): 
        ax.scatter(wn[picos], curva_grafico[picos], c='#E94A5C', zorder=5, s=60, label="Absorciones Detectadas")

    ax.set_xlabel("Número de onda (cm⁻¹)", fontsize=11, fontweight='bold', color='#2d3436')
    ax.set_ylabel("Transmitancia Relativa", fontsize=11, fontweight='bold', color='#2d3436')
    ax.set_title("Señal Infrarroja Extraída", fontsize=13, fontweight='bold', color='#2d3436', pad=12)
    ax.invert_xaxis()
    ax.grid(color='#b2bec3', linestyle=':', linewidth=1)
    ax.legend(facecolor='#ffffff', edgecolor='#dfe6e9')
    fig.tight_layout()

    identificacion_texto = identificar_compuesto(resultados)

    return fig, resultados, identificacion_texto

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero-box">
  <h1 class="hero-title">SpectroAnalyzer PRO</h1>
  <p style="color: #636e72; font-size: 1.1rem; margin-top: 5px;">Plataforma Avanzada de Espectroscopía Infrarroja (IR) con Visión Artificial</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PESTAÑAS ----------------
tab1, tab2, tab3 = st.tabs(["🔬 Analizador Inteligente", "📂 Base de Datos de Referencia", "📖 Fundamentos Teóricos"])

with tab1:
    st.markdown("### Motor de Inferencia Computacional")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        uploaded_file = st.file_uploader("Sube un espectro IR (PNG, JPG, JPEG, WEBP)", type=["png", "jpg", "jpeg", "webp"])
        
        c_left, c_right = st.columns(2)
        with c_left:
            wn_izq = st.number_input("Límite X Izquierdo (cm⁻¹)", value=4000)
        with c_right:
            wn_der = st.number_input("Límite X Derecho (cm⁻¹)", value=500)

        with st.expander("⚙️ Calibración de Recorte Visual"):
            sensibilidad = st.slider("Sensibilidad de Picos", 1, 30, 5)
            x_inicio = st.slider("Recorte Vertical Izquierdo (%)", 0, 45, 6)
            x_fin = st.slider("Recorte Vertical Derecho (%)", 55, 100, 90)
            y_inicio = st.slider("Recorte Horizontal Superior (%)", 0, 45, 15)
            y_fin = st.slider("Recorte Horizontal Inferior (%)", 55, 100, 85)

    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            fig, resultados, identificacion_texto = analizar_imagen_ir(
                image, wn_izq, wn_der, sensibilidad, x_inicio, x_fin, y_inicio, y_fin
            )
            
            if fig is not None:
                st.markdown(identificacion_texto, unsafe_allow_html=True)
                st.pyplot(fig)
                
                if resultados:
                    st.markdown("#### 📊 Tabla de Absorciones Detectadas")
                    tabla_md = "| # Pico | Frecuencia (cm⁻¹) |\n|---:|---:|\n"
                    for i, r in enumerate(resultados):
                        tabla_md += f"| {i+1} | **{r['wn']:.1f}** |\n"
                    st.markdown(tabla_md)
        else:
            st.info("👈 Sube una imagen de un espectro IR a la izquierda para iniciar el análisis.")

with tab2:
    st.markdown("### Repositorio de Firmas Espectrales")
    grupo_sel = st.selectbox("Seleccionar Familia Química", list(GRUPOS.keys()))
    info = GRUPOS[grupo_sel]
    
    st.markdown(f"<div class='info-card'>{info['desc']}</div>", unsafe_allow_html=True)
    
    carpeta = os.path.join(BASE_DIR, info["folder"])
    encontrados = []
    if os.path.exists(carpeta):
        for ext in EXTENSIONES:
            encontrados.extend(glob.glob(os.path.join(carpeta, ext)))
            encontrados.extend(glob.glob(os.path.join(carpeta, ext.upper())))
    
    encontrados = sorted(set(encontrados))
    
    if encontrados:
        st.write(f"Se encontraron **{len(encontrados)}** imágenes de ejemplo:")
        cols = st.columns(3)
        for idx, img_path in enumerate(encontrados):
            with cols[idx % 3]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.warning(f"No hay imágenes guardadas en la carpeta `{carpeta}`.")

with tab3:
    st.markdown("""
    ## 🔬 Principios de la Espectroscopía Infrarroja (IR)
    
    La espectroscopía infrarroja mide las transiciones vibracionales de las moléculas. Cuando una molécula absorbe radiación infrarroja, sus enlaces vibran con mayor amplitud.
    
    ### 1. Requisito Fundamental: El Momento Dipolar
    Para que un enlace absorba energía en el infrarrojo, **su vibración debe causar un cambio en el momento dipolar** de la molécula. Moléculas simétricas son "invisibles" en el espectro IR.
    
    ### 2. Modelo del Resorte (Ley de Hooke)
    La frecuencia de vibración depende de:
    * **La fuerza del enlace ($k$):** Enlaces más fuertes vibran a frecuencias más altas.
    * **La masa de los átomos ($\mu$):** Átomos más ligeros (como el Hidrógeno) vibran a frecuencias más altas (~3000 cm⁻¹).
    
    ### 3. Absorciones Clave
    | Grupo Funcional | Región (cm⁻¹) | Características Visuales |
    | :--- | :--- | :--- |
    | **O-H (Alcohol)** | 3200 - 3600 | Banda fuerte, muy ancha en forma de lengua. |
    | **O-H (Ácido)** | 2500 - 3300 | Banda gigantesca y deforme que pisa a los C-H. |
    | **N-H (Amina)** | 3300 - 3500 | Picos de intensidad media, algo afilados. |
    | **C-H (Alifático)** | 2850 - 3000 | Inmediatamente a la derecha de la línea de 3000. |
    | **C-H (Aromático/Alqueno)**| 3000 - 3150 | Inmediatamente a la izquierda de la línea de 3000. |
    | **C≡C / C≡N** | 2100 - 2260 | Picos débiles a medios en una zona usualmente vacía. |
    | **C=O (Carbonilo)** | 1670 - 1850 | El pico más intenso, agudo y profundo del espectro. |
    | **C=C (Doble enlace)** | 1620 - 1680 | Pico medio o débil. |
    """)
