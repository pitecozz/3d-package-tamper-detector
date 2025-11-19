import gradio as gr
import numpy as np
import cv2
import os
from src.core.detector import PackageTamperDetector # Importa o orquestrador core
from PIL import Image

# Inicializa o Detector (o modelo de profundidade será carregado aqui)
try:
    detector = PackageTamperDetector()
except Exception as e:
    # Se houver erro no carregamento do modelo HF (GPU/CPU), usa um mock
    print(f"Erro ao inicializar o detector principal: {e}. Usando mock.")
    class MockDetector:
        def analyze_video(self, video_path):
            return {
                'is_tampered': True,
                'interpretation': "🚨 MOCK: Falha na Geometria. Violação Detectada.",
                'visual_data': {'frame_rgb': np.zeros((480, 640, 3), dtype=np.uint8), 'depth_map': np.zeros((480, 640))}
            }
    detector = MockDetector()

def analyze_package(video_path):
    """Função wrapper para o Gradio."""
    if video_path is None:
        return "Por favor, faça upload de um vídeo.", None
            
    # Executa o pipeline de análise
    results = detector.analyze_video(video_path)
    
    # Prepara as visualizações para o Gradio
    frame_rgb = results['visual_data']['frame_rgb']
    depth_map = results['visual_data']['depth_map']
    
    # Converte o depth map para visualização colorida (Jet Colormap)
    depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    # Redimensiona o frame original para ter a mesma altura da visualização de profundidade (se necessário)
    depth_vis_resized = cv2.resize(depth_vis, (frame_rgb.shape[1], frame_rgb.shape[0]))

    # Combinar o frame original e o mapa de profundidade lado a lado
    combined_vis = np.hstack([frame_rgb, depth_vis_resized])
    
    # Cria um placeholder para o gráfico de nuvem de pontos (que o Gradio não suporta diretamente)
    pcd_placeholder = Image.fromarray(combined_vis)

    return results['interpretation'], pcd_placeholder

iface = gr.Interface(
    fn=analyze_package,
    inputs=gr.Video(label="Envie o Vídeo do Pacote (360°) 📦", sources=["upload"]),
    outputs=[
        gr.Textbox(label="Status de Integridade 🛡️"),
        gr.Image(label="Visualização 3D (Frame Original + Mapa de Profundidade)", type="pil")
    ],
    title="3D Package Tamper Detector (iFood GenAI Project)",
    description="""
    🔍 **Detector de Violação de Pacotes usando Reconstrução Geométrica 3D (MVP)**
    
    Este projeto demonstra como a IA, inspirada em técnicas de visão 3D (Depth Estimation), pode analisar a geometria de um pacote em movimento para detectar violações (cortes, amassados) que comprometem a integridade do produto.
    
    **Sinal para iFood:** Aplicação de Visão/GenAI em Segurança Logística.
    """
)

iface.launch()
