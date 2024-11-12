import torch
import torch.nn.functional as F
import numpy as np
import cv2

def get_rotation_matrix_tensor(rad, ax):
    ax = torch.tensor(ax, device='cuda')
    assert len(ax.shape) == 1 and ax.shape[0] == 3
    ax = ax / torch.sqrt((ax**2).sum())
    R = torch.diag(torch.cos(rad) * torch.ones(3, device='cuda'))
    R = R + torch.outer(ax, ax) * (1.0 - torch.cos(rad))

    ax = ax * torch.sin(rad)
    R = R + torch.tensor([[0, -ax[2], ax[1]], 
                         [ax[2], 0, -ax[0]], 
                         [-ax[1], ax[0], 0]], device='cuda')
    return R

def grid_in_3d_to_project_tensor(o_fov, o_sz, o_u, o_v):
    z = torch.tensor(1.0, device='cuda')
    L = torch.tan(o_fov / 2) / z
    x = torch.linspace(L, -L, o_sz, device='cuda')
    y = torch.linspace(-L, L, o_sz, device='cuda')
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    z_grid = torch.ones_like(x_grid, device='cuda')

    Rx = get_rotation_matrix_tensor(o_v, [1, 0, 0])
    Ry = get_rotation_matrix_tensor(o_u, [0, 1, 0])
    xyz_grid = torch.stack([x_grid, y_grid, z_grid], -1)
    xyz_grid = xyz_grid @ Rx @ Ry

    return [xyz_grid[..., i] for i in range(3)]

def fisheye_to_plane_info_tensor(frame, ih, iw, i_fov=180, o_fov=90, o_sz=600, o_u=0, o_v=0):
    """
    Args:
        frame: [H, W, C] 형태의 GPU 텐서
    """
    i_fov = torch.tensor(i_fov * np.pi / 180, device='cuda')
    o_fov = torch.tensor(o_fov * np.pi / 180, device='cuda')
    o_u = torch.tensor(o_u * np.pi / 180, device='cuda')
    o_v = torch.tensor(o_v * np.pi / 180, device='cuda')

    x_grid, y_grid, z_grid = grid_in_3d_to_project_tensor(o_fov, o_sz, o_u, o_v)

    theta = torch.atan2(y_grid, x_grid)
    c_grid = torch.sqrt(x_grid**2 + y_grid**2)
    rho = torch.atan2(c_grid, z_grid)
    r = rho * min(ih, iw) / i_fov

    coor_x = r * torch.cos(theta) + iw / 2
    coor_y = r * torch.sin(theta) + ih / 2

    coor_x = (coor_x / (iw - 1)) * 2 - 1
    coor_y = (coor_y / (ih - 1)) * 2 - 1

    grid = torch.stack([coor_x, coor_y], dim=-1).unsqueeze(0)
    frame = frame.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    transformed_image = F.grid_sample(
        frame, 
        grid, 
        mode='bilinear', 
        align_corners=True,
        padding_mode='zeros'
    )

    transformed_image = transformed_image.squeeze(0).permute(1, 2, 0)
    

    transformed_image = torch.flip(transformed_image, [1])

    return transformed_image

def run(frame, view, move=0, flag=True, cam_name=None):
    """
    Args:
        frame: [H, W, C] 형태의 GPU 텐서
    """
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame).to('cuda')
        frame = frame.to(torch.float32)
        
    h, w = frame.shape[:2]

    black = torch.zeros((int((w - h) // 2), w, 3), dtype=frame.dtype, device='cuda')
    frame_new = torch.cat([black, frame, black], dim=0)
    
    h_new, w_new = frame_new.shape[:2]
    
    res = fisheye_to_plane_info_tensor(
        frame_new, h_new, w_new, 180, 90, 640, view, move
    )
    
    if flag:
        res = torch.rot90(res, k=1, dims=[0, 1])
    else:
        res = torch.rot90(res, k=-1, dims=[0, 1])
    
    if cam_name == 'cam2_2':
        res[:120, :, :] = 0
    
    if isinstance(res, torch.Tensor):
        return (res).to(torch.uint8).cpu().numpy()
    else:
        return res

if __name__ == "__main__":
    import numpy as np
    import cv2
    import time
    import logging
    current_time = time.time()
    frame = cv2.imread('collect/scne2_0.5origin/cam2/129.jpg')
    result = run(frame, -40, 0, True, 'cam2')
    print(f"처리 시간: {time.time() - current_time:.4f} seconds")
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
