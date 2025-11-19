import numpy as np
from sklearn.ensemble import IsolationForest
import open3d as o3d
from sklearn.preprocessing import StandardScaler

class TamperDetector:
    """
    Detector de Violação baseado em Análise Geométrica 3D.
    Usa Isolation Forest com Normalização de Features.
    """
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(
            contamination=0.15,  # Aumenta a tolerância para 15% (menos falsos positivos)
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_geometry_features(self, point_cloud):
        """
        Calcula features geométricas essenciais (Volume, Rugosidade, Pontos).
        (ESTA É A FUNÇÃO QUE ESTAVA FALTANDO E CAUSANDO O ATTRIBUTEERROR)
        """
        points = np.asarray(point_cloud.points)
        
        if len(points) < 50:
            return np.array([0.0, 100.0, 0]).reshape(1, -1)
        
        # 1. Volume Estimado
        bbox = point_cloud.get_axis_aligned_bounding_box()
        volume = bbox.volume()
        
        # 2. Rugosidade/Variação da Superfície
        center = np.mean(points, axis=0)
        distance_variance = np.var(np.linalg.norm(points - center, axis=1))
        
        # 3. Número de Pontos
        num_points = len(points)
        
        features = np.array([
            volume, 
            distance_variance, 
            num_points
        ]).reshape(1, -1)
        return features

    def train_on_normal_data(self, normal_data_features):
        """Treina o detector de anomalias com um conjunto de features de pacotes normais."""
        self.scaler.fit(normal_data_features)
        scaled_features = self.scaler.transform(normal_data_features)
        self.anomaly_detector.fit(scaled_features)
        self.is_trained = True

    def predict_tamper(self, features):
        """Prevê se as features são anômalas (violadas)."""
        if not self.is_trained:
            return 1 
        
        scaled_features = self.scaler.transform(features)
        return self.anomaly_detector.predict(scaled_features)[0]
