import numpy as np
from sklearn.ensemble import IsolationForest
import open3d as o3d

class TamperDetector:
    """
    Detector de Violação baseado em Análise Geométrica 3D (Isolation Forest).
    Treina em pacotes 'normais' e detecta qualquer desvio anômalo.
    """
    
    def __init__(self):
        # Isolation Forest: Excelente para Anomaly Detection, ideal para detectar 
        # a "assinatura" geométrica de uma violação (que é um outlier raro).
        self.anomaly_detector = IsolationForest(
            contamination=0.05,  # Assume que 5% dos dados iniciais podem ser outliers
            random_state=42
        )
        self.is_trained = False

    def extract_geometry_features(self, point_cloud):
        """Calcula features geométricas essenciais para detecção de violação."""
        points = np.asarray(point_cloud.points)
        
        if len(points) < 50:
             # Retorna um conjunto de features que forçará um alerta se os dados forem ruins
            return np.array([0.0, 100.0, 0]).reshape(1, -1)
        
        # 1. Volume Estimado (Bounding Box): Uma violação severa pode reduzir o volume
        bbox = point_cloud.get_axis_aligned_bounding_box()
        volume = bbox.volume()
        
        # 2. Rugosidade/Variação da Superfície: Amassados/cortes aumentam a rugosidade.
        # Calculamos a variância da distância dos pontos ao centroide.
        center = np.mean(points, axis=0)
        distance_variance = np.var(np.linalg.norm(points - center, axis=1))
        
        # 3. Número de Pontos: A remoção de conteúdo ou má reconstrução pode alterar isso.
        num_points = len(points)
        
        features = np.array([
            volume, 
            distance_variance, 
            num_points
        ]).reshape(1, -1)
        return features

    def train_on_normal_data(self, normal_data_features):
        """Treina o detector de anomalias com um conjunto de features de pacotes normais."""
        self.anomaly_detector.fit(normal_data_features)
        self.is_trained = True
        
    def predict_tamper(self, features):
        """Prevê se as features são anômalas (violadas)."""
        if not self.is_trained:
            # Em um cenário real, isso lançaria um erro; para o MVP, assume que o treino simulado ocorreu
            return 1 # Assume normal se não treinado
            
        # prediction -1: Anomalia (Violado), prediction 1: Normal (Íntegro)
        return self.anomaly_detector.predict(features)[0]
