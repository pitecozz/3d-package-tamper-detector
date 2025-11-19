import sys
import os
import gradio as gr
import numpy as np
import cv2
from PIL import Image

# Corrigido para importação robusta
from src.vision.visualizer import create_depth_visualization
from src.core.detector import PackageTamperDetector

# 1. CORREÇÃO CRÍTICA DO PATH (Final e Robusta)
# Injeta a pasta raiz do projeto no sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. INICIALIZAÇÃO DO DETECTOR
try:
    detector = PackageTamperDetector()
except Exception as e:
    # Fallback/Mock para garantir que a interface não quebre
    print(f"Erro CRÍTICO ao inicializar o detector (falha de recurso): {e}. Usando Mock.")
    class MockDetector:
        def analyze_video(self, video_path):
            return {
                'is_tampered': True,
                'interpretation': "🚨 FALHA NA INICIALIZAÇÃO: O detector está no modo Mock. Violação Detectada.",
                'visual_data': {'frame_rgb': np.zeros((480, 640, 3), dtype=np.uint8), 'depth_map': np.zeros((480, 640))}
            }
    detector = MockDetector()

def analyze_package(video_path):
    """Função principal chamada pelo Gradio."""
    if video_path is None:
        return "Por favor, faça upload de um vídeo do pacote para iniciar a análise.", None 
    
    # Executa o pipeline de análise
    results = detector.analyze_video(video_path)

    # Prepara as visualizações de alta qualidade
    frame_rgb = results['visual_data']['frame_rgb']
    depth_map = results['visual_data']['depth_map']
    
    # 1. CRÍTICO: Cria a visualização de profundidade detalhada (Matplotlib)
    depth_vis_pil = create_depth_visualization(depth_map)
    
    # 2. Converte e redimensiona a imagem para combinar
    frame_pil = Image.fromarray(frame_rgb)
    
    # Garante que o mapa de profundidade tem a mesma altura e combina as imagens
    if depth_vis_pil is not None:
        depth_vis_resized = depth_vis_pil.resize(frame_pil.size)
        
        combined_vis = Image.new('RGB', (frame_pil.width + depth_vis_resized.width, frame_pil.height))
        combined_vis.paste(frame_pil, (0, 0))
        combined_vis.paste(depth_vis_resized, (frame_pil.width, 0))
        
        return results['interpretation'], combined_vis
    else:
        # Fallback se a visualização falhar
        return "Análise concluída, mas a visualização falhou.", frame_pil 

# DEFINIÇÃO DA INTERFACE GRADIO
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
