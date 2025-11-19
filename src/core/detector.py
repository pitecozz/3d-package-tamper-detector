import numpy as np
import open3d as o3d
import os
import cv2 

# Importando os módulos implementados
from src.vision.depth_estimator import DepthEstimator
from src.geometry.reconstructor import MultiViewReconstructor
from src.geometry.tamper_detector import TamperDetector

# --- Funções de Simulação Otimizada ---

def _simulate_training_data(n_samples=200):
    """
    Gera features sintéticas com MAIS VARIAÇÃO e MAIS DADOS (200 amostras)
    para um treinamento robusto, corrigindo a alta sensibilidade.
    """
    np.random.seed(42)
    
    # Simula Variações Realistas para o 'Normal'
    # Volume: pacotes pequenos a médios, com grande amplitude de variação
    sim_volume = np.random.uniform(0.001, 0.20, n_samples) 
    # Variância (Rugosidade): tolerância a superfícies irregulares (sacolas)
    sim_variance = np.random.uniform(0.5, 10.0, n_samples) 
    # Número de pontos (densidade): boa variação
    sim_num_points = np.random.randint(5000, 80000, n_samples) 

    return np.column_stack([sim_volume, sim_variance, sim_num_points])

class PackageTamperDetector:
    """
    Orquestrador principal do sistema de detecção de violação 3D.
    """
    
    def __init__(self):
        # Inicializa todos os subsistemas
        self.depth_estimator = DepthEstimator()
        self.reconstructor = MultiViewReconstructor()
        self.tamper_detector = TamperDetector()
        
        # Treina o detector na inicialização (MVP com dados simulados)
        self._simulate_training_and_fit()

    def _simulate_training_and_fit(self):
        """Prepara o detector de anomalias."""
        print("[Detector] Simulating training on 200 normal packages...")
        normal_features = _simulate_training_data()
        self.tamper_detector.train_on_normal_data(normal_features)
        print("[Detector] Training complete. Detector ready.")
        
    def analyze_video(self, video_path):
        """
        Executa o pipeline completo: Visão -> 3D -> Análise Geométrica.
        """
        # Verifica se o arquivo existe (necessário para rodar no servidor)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        # 1. VISÃO: Extrair Frames (Imagens RGB) e Depth Maps (Profundidade)
        print("[Pipeline] Extracting frames and estimating depth...")
        frames = self.depth_estimator.extract_frames(video_path, num_frames=8)
        
        if not frames:
            # Fallback robusto
            return {'is_tampered': False, 'interpretation': "Erro de leitura do vídeo.", 'visual_data': {'frame_rgb': np.zeros((480, 640, 3), dtype=np.uint8), 'depth_map': np.zeros((480, 640))}}

        depth_maps = [self.depth_estimator.estimate_depth(f) for f in frames]
        
        # 2. GEOMETRIA: Reconstrução 3D
        pcds = [self.reconstructor.to_point_cloud(f, d) for f, d in zip(frames, depth_maps)]
        pcd_merged = self.reconstructor.align_and_merge(pcds)
        print(f"[Pipeline] 3D Point Cloud created ({len(pcd_merged.points)} points).")
        
        # 3. SEGURANÇA: Extrair Features e Prever Violação
        # CORREÇÃO: Chama o método diretamente do detector e passa a nuvem de pontos
        features = self.tamper_detector.extract_geometry_features(pcd_merged)
        prediction = self.tamper_detector.predict_tamper(features)
        
        # Interpretação
        is_tampered = prediction == -1
        
        return {
            'is_tampered': is_tampered,
            'confidence_score': "N/A (Isolation Forest)",
            'interpretation': "🚨 VIOLAÇÃO DETECTADA (Anomalia Geométrica)" if is_tampered else "✅ PACOTE ÍNTEGRO (Geometria Normal)",
            'visual_data': {
                'frame_rgb': frames[0],
                'depth_map': depth_maps[0],
                'point_count': len(pcd_merged.points)
            }
        }
