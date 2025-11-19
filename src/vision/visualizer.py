import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import cv2

def create_depth_visualization(depth_map):
    """
    Cria um mapa de profundidade detalhado usando Matplotlib, ajustando a escala
    para focar apenas no pacote (ignorando o ruído de fundo).
    """
    
    # 1. Normalização dos valores de profundidade para o colormap
    valid_depths = depth_map[depth_map > 0] 
    
    if valid_depths.size == 0:
        return Image.fromarray(np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8))

    # Ajusta os percentis para ter uma escala de cores mais sensível ao objeto
    d_min = np.percentile(valid_depths, 5) 
    d_max = np.percentile(valid_depths, 95)
    
    if d_max == d_min:
        d_max = d_min + 1e-6

    # 2. Criação do Gráfico Matplotlib (Melhor qualidade visual)
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Usa 'viridis' (mais informativa) e define a escala de cor (vmin, vmax)
    ax.imshow(depth_map, cmap='viridis', vmin=d_min, vmax=d_max) 
    
    ax.axis('off') # Remove eixos e bordas
    ax.margins(0, 0)
    fig.tight_layout(pad=0)
    
    # 3. Captura e Conversão para PIL Image (Necessário para combinar no Gradio)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    img = Image.open(buf).convert("RGB")
    return img
