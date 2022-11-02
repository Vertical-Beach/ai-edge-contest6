import copy
import numpy as np
from pyquaternion import Quaternion


class LidarPointCloud():
    def __init__(self, points):
        self.points = points

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

def lidar_to_global(points,
                    lidar_ego_pose,
                    lidar_calibration):
    """
    Lidar座標 -> Global座標 に変換

    Args:
        points: Lidar座標の点群 shape = (N, C)
        lidar_ego_pose: Lidarのegoポーズパラメータ
       lidar_calibration: Lidarのキャリブレーションパラメータ
    Returns:
        points_global: Glabal座標の点群 shape = (N, C)
    """

    points_tmp = copy.deepcopy(points)
    num_point_feature = points_tmp.shape[1]
    assert num_point_feature >= 3, f"num_point_feature: {num_point_feature} は3以上必要です。"

    if num_point_feature == 3:  # (N, 3) -> (N, 4)
        points_tmp = np.hstack([points_tmp, np.zeros([points_tmp.shape[0], 1])])

    elif num_point_feature >= 4:
        points_feature = copy.deepcopy(points[:, 3:])
        points_tmp = points_tmp[:, :4]

    pc = LidarPointCloud(points_tmp.T)

    # Lidar座標系 -> カメラ座標系 に変換
    # 1st step: Lidar座標 -> Lidar車両ポーズ
    pc.rotate(Quaternion(lidar_calibration['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibration['translation']))

    # 2nd step: Lidar車両ポーズ -> Global座標
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    points_global = pc.points[:3, :].T        # (N, 3)  N:(x, y, z)
    if num_point_feature >= 4:
        points_global = np.hstack([points_global, points_feature])

    return points_global