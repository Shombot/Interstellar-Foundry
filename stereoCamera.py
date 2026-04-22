import numpy as np


def backproject_bbox_center_to_3d(bbox, depth_m, fx, fy, cx, cy):
    """
    bbox: (x1, y1, x2, y2) in pixels
    depth_m: depth at bbox center in meters
    returns: np.array([X, Y, Z]) in camera frame, or None
    """
    if depth_m is None or depth_m <= 0:
        return None

    x1, y1, x2, y2 = bbox
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)

    X = depth_m * (u - cx) / fx
    Y = depth_m * (v - cy) / fy
    Z = depth_m
    return np.array([X, Y, Z], dtype=np.float32)


def transform_radar_to_camera(p_radar, R_CR, t_CR):
    """
    p_radar: np.array([x_r, y_r, z_r])
    R_CR: 3x3 rotation
    t_CR: 3-vector
    """
    return R_CR @ p_radar + t_CR


def project_camera_point_to_image(p_cam, fx, fy, cx, cy):
    """
    p_cam: np.array([X, Y, Z]) in camera frame
    returns: (u, v) or None if behind camera
    """
    X, Y, Z = p_cam
    if Z <= 1e-6:
        return None
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return (float(u), float(v))


def point_in_bbox(u, v, bbox, margin=0):
    x1, y1, x2, y2 = bbox
    return (x1 - margin) <= u <= (x2 + margin) and (y1 - margin) <= v <= (y2 + margin)


def association_cost(det_bbox, det_depth_m, radar_p_cam, fx, fy, cx, cy,
                     sigma_u=60.0, sigma_v=60.0, sigma_z=2.0):
    """
    Lower cost = better match
    """
    uv = project_camera_point_to_image(radar_p_cam, fx, fy, cx, cy)
    if uv is None:
        return np.inf

    u_r, v_r = uv
    x1, y1, x2, y2 = det_bbox
    u_d = 0.5 * (x1 + x2)
    v_d = 0.5 * (y1 + y2)
    z_r = radar_p_cam[2]

    cost = ((u_d - u_r) ** 2) / (sigma_u ** 2) + ((v_d - v_r) ** 2) / (sigma_v ** 2)

    if det_depth_m is not None and det_depth_m > 0:
        cost += ((det_depth_m - z_r) ** 2) / (sigma_z ** 2)

    return cost


def fuse_positions_weighted(p_cam_meas, p_rad_meas, R_cam, R_rad):
    """
    Weighted least-squares fusion of two 3D position measurements.
    """
    Wc = np.linalg.inv(R_cam)
    Wr = np.linalg.inv(R_rad)
    P = np.linalg.inv(Wc + Wr)
    x = P @ (Wc @ p_cam_meas + Wr @ p_rad_meas)
    return x, P