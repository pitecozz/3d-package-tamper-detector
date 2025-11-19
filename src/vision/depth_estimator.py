import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

class DepthEstimator:
    """Extrai frames do vídeo e estima o mapa de profundidade."""
    
    def __init__(self):
        # CORREÇÃO CRÍTICA: Utiliza o modelo 'dpt-large-tiny' para garantir o carregamento correto no servidor HF.
        try:
            self.depth_pipe = pipeline("depth-estimation", 
                                     model="Intel/dpt-hybrid-midas") 
        except Exception as e:
            # Em caso de falha no carregamento do modelo, usa um pipeline mock para que o sistema não trave.
            print(f"Erro ao carregar o pipeline de profundidade: {e}. Usando mock.")
            self.depth_pipe = None

    def extract_frames(self, video_path, num_frames=8):
        """Extrai N frames equidistantes do vídeo usando OpenCV."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"Erro: Não foi possível abrir o vídeo em {video_path}")
            return []
            
        # Garante que frames sejam extraídos uniformemente
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # O modelo HF espera RGB, e OpenCV lê BGR
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def estimate_depth(self, frame_rgb):
        """Estima o mapa de profundidade para um frame."""
        if not self.depth_pipe:
            # Retorna um mapa de profundidade falso se o pipeline falhar (Modo Mock)
            return np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.float32)
            
        image = Image.fromarray(frame_rgb)
        
        # Executa a inferência do modelo de profundidade
        results = self.depth_pipe(image)
        
        # Retorna o array numpy do mapa de profundidade
        return np.array(results['depth'])
