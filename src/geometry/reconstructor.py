import numpy as np
import open3d as o3d

class MultiViewReconstructor:
    """
    Cria a Nuvem de Pontos 3D a partir de Múltiplas Vistas (Frames/Depth Maps).
    Utiliza o princípio RGBD (Cor + Profundidade) para a reconstrução geométrica,
    inspirado no poder de percepção espacial do Depth Anything.
    """
    
    def __init__(self, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
        # Parâmetros Intrínsecos Simplificados para uma resolução 640x480 (comum em vídeos)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, fx, fy, cx, cy
        )

    def to_point_cloud(self, frame_rgb, depth_map):
        """Converte frame RGB e Depth Map em uma Nuvem de Pontos Open3D. """
        
        # 1. Preparação dos dados para Open3D
        # Open3D espera que a imagem de profundidade esteja no formato float32
        depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
        image_o3d = o3d.geometry.Image(frame_rgb.astype(np.uint8))

        # 2. Criação da Imagem RGBD (Profundidade + Cor)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            image_o3d, depth_o3d, 
            depth_scale=1000.0, # Assumindo profundidade em milímetros
            depth_trunc=3.0,    # Ignora pontos acima de 3 metros (foco no pacote)
            convert_rgb_to_intensity=False
        )
        
        # 3. Criação da Nuvem de Pontos (PointCloud)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.intrinsic
        )
        return pcd

    def align_and_merge(self, pcds):
        """Alinha e mescla múltiplas nuvens de pontos (força bruta para MVP)."""
        
        final_pcd = o3d.geometry.PointCloud()
        
        # Merge simples: soma todas as nuvens. Aplicamos uma pequena translação
        # para simular a rotação da câmera e evitar que todos os pontos fiquem
        # exatamente no mesmo plano de projeção, melhorando a densidade 3D.
        for i, pcd in enumerate(pcds):
            # Translação artificial para ajudar o alinhamento
            pcd.translate((0.005 * i, 0.0, 0.0)) 
            final_pcd += pcd 

        # Otimização: Downsampling (reduz o número de pontos para performance)
        final_pcd = final_pcd.voxel_down_sample(voxel_size=0.005) 
        
        # Remoção de Ruído (Outliers)
        final_pcd, ind = final_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return final_pcd
