import cv2
import natsort.natsort
import numpy as np
import time



def get_rotation_matrix(rad, ax):
    ax = np.array(ax)
    assert len(ax.shape) == 1 and ax.shape[0] == 3
    ax = ax / np.sqrt((ax**2).sum())
    R = np.diag([np.cos(rad)] * 3)
    R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))

    ax = ax * np.sin(rad)
    R = R + np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return R


def grid_in_3d_to_project(o_fov, o_sz, o_u, o_v):

    z = 1
    L = np.tan(o_fov / 2) / z
    x = np.linspace(L, -L, num=o_sz, dtype=np.float64)
    y = np.linspace(-L, L, num=o_sz, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.ones_like(x_grid)

    Rx = get_rotation_matrix(o_v, [1, 0, 0])
    Ry = get_rotation_matrix(o_u, [0, 1, 0])
    xyz_grid = np.stack([x_grid, y_grid, z_grid], -1).dot(Rx).dot(Ry)

    return [xyz_grid[..., i] for i in range(3)]


def fisheye_to_plane_info(frame, ih, iw, i_fov=180, o_fov=90, o_sz=600, o_u=0, o_v=0):

    # [i_fov, o_fov, o_sz, o_u, o_v]
    # i_fov: input fov (180도 카메라이므로 180으로 고정),
    # o_sz: 해상도 설정
    # o_fov: output fov   배율  (zoom in  / zoom out)
    # o_u: move right(왼쪽으로 돌릴 각도 크기), o_v: move down(아래로 돌릴 각도 크기) -> 범위 -90 ~ 90
    i_fov = i_fov * np.pi / 180
    o_fov = o_fov * np.pi / 180
    o_u = o_u * np.pi / 180
    o_v = o_v * np.pi / 180

    x_grid, y_grid, z_grid = grid_in_3d_to_project(o_fov, o_sz, o_u, o_v)

    theta = np.arctan2(y_grid, x_grid)
    c_grid = np.sqrt(x_grid**2 + y_grid**2)
    rho = np.arctan2(c_grid, z_grid)
    r = rho * min(ih, iw) / i_fov
    coor_x = r * np.cos(theta) + iw / 2
    coor_y = r * np.sin(theta) + ih / 2

    transformed_image = cv2.remap(
        frame,
        coor_x.astype(np.float32),
        coor_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
    )

    out = np.fliplr(transformed_image)

    return out

def run(frame , view , flag = True):
    """_summary_

    Args:
        frame (_type_): target image
        view (_type_): "위 아래 카메라 각도"
        fag (_type_) : True -> ROTATE_90_COUNTERCLOCKWISE
    """
    #frame = cv2.imread(frame , cv2.IMREAD_COLOR)

    h, w, _ = list(map(int, frame.shape))

    black = np.zeros(((int(w - h) // 2), w, 3), dtype=np.uint8)

    frame_new = cv2.vconcat([black, frame])
    frame_new = cv2.vconcat([frame_new, black])

    h, w, _ = list(map(int, frame_new.shape))
    
    res = np.array(fisheye_to_plane_info(frame_new , h , w , 180 , 90 , 640 , view , 0))
    if flag:
        res = cv2.rotate(res , cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        res = cv2.rotate(res , cv2.ROTATE_90_CLOCKWISE)
    return res

def adjust(f_list , u_list):

    zip_list = list(zip(f_list , u_list))
    for d , u  in zip_list:
        frame = cv2.imread(d , cv2.IMREAD_COLOR)
        h, w, _ = list(map(int, frame.shape))

        black = np.zeros(((int(w - h) // 2), w, 3), dtype=np.uint8)

        frame_new = cv2.vconcat([black, frame])
        frame_new = cv2.vconcat([frame_new, black])

        h, w, _ = list(map(int, frame_new.shape))

        t1 = np.array(fisheye_to_plane_info(frame_new, h, w, 180, 90, 600, -40, 0))

        frame_new = cv2.imread(u , cv2.IMREAD_COLOR)
        h, w, _ = list(map(int, frame_new.shape))

        black = np.zeros(((int(w - h) // 2), w, 3), dtype=np.uint8)

        frame_new = cv2.vconcat([black, frame])
        frame_new = cv2.vconcat([frame_new, black])

        h, w, _ = list(map(int, frame_new.shape))

        t2 = np.array(fisheye_to_plane_info(frame_u , h , w , 180 , 90 , 600 , -45 , 0))

        cv2.namedWindow("t1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("t2" , cv2.WINDOW_NORMAL)
        cv2.imshow("t1", t1)
        cv2.imshow("t2",t2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import natsort
    import glob
    import os

    door_list = natsort.natsorted(glob.glob(os.path.join('SCNE1/door1','*.jpg')))
    under_list = natsort.natsorted(glob.glob(os.path.join('SCNE1/under','*.jpg')))


    h, w, frame_new = adjust(door_list)
    h_u , w_u , frame_u = adjust(under_list)

    t1 = np.array(fisheye_to_plane_info(frame_new, h, w, 180, 90, 600, -40, 0))
    t2 = np.array(fisheye_to_plane_info(frame_u , h , w , 180 , 90 , 600 , -45 , 0))
    t1 = cv2.rotate(t1 , cv2.ROTATE_90_COUNTERCLOCKWISE)
    

    cv2.namedWindow("t1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("t2" , cv2.WINDOW_NORMAL)
    cv2.imshow("t1", t1)
    cv2.imshow("t2",t2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
