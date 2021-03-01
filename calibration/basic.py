import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit


def line(ax, p1, p2, *args, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], *args, **kwargs)


def basis(ax, T, R, *args, length=1, **kwargs):
    line(ax, T, T + length * R[0, :], "r")
    line(ax, T, T + length * R[1, :], "g")
    line(ax, T, T + length * R[2, :], "b")


def field(ax, dim_x, dim_y, length=30, *args, label="field", **kwargs):
    o, ex, ey = np.zeros(3), np.array([1, 0, 0]), np.array([0, 1, 0])

    line(ax, o, o + ex * dim_x, "orange", linestyle="--", label=label)
    line(ax, o, o + ey * dim_y, "orange", linestyle="--")
    line(ax, o + ex * dim_x, o + ex * dim_x + o + ey * dim_y, "orange", linestyle="--")
    line(ax, o + ey * dim_y, o + ex * dim_x + o + ey * dim_y, "orange", linestyle="--")

    basis(ax, np.ones(3), np.eye(3), length=length)


def axis_equal_3d(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def cache_reference(filename, plot=False):
    ref = cv2.imread(filename + ".png")[:, :, ::-1]
    print(ref.shape, ref.dtype)

    if plot:
        plt.figure("Color Reference")
        plt.imshow(ref[::10, ::10, :])

    ref[np.nonzero(ref)] = 1
    ref *= np.array([1, 2, 3], dtype=np.uint8)[None, None, :]

    gray = np.zeros(ref.shape[:2], dtype=np.uint8)
    gray[...] = np.sum(ref, axis=2, dtype=np.uint8)
    gray[np.equal(gray, 6)] = 4

    np.save(filename + ".npy", gray)


def load_reference(filename, plot=False):
    ref = np.load(reference_filename + ".npy")
    print(ref.shape, ref.dtype)

    if plot:
        plt.figure("Gray Reference")
        plt.imshow(ref[::10, ::10])
        plt.colorbar()

    return ref


height, width = 1080*2, 1920*2

# ref_dim and ref_origin in feet (x-right, y-down, top-left image origin)
#
def render(ref, *calib, ref_dim=(430, 430), ref_origin=(355, 75), plot=False):
    print(calib)

    if len(ref.shape) == 1:
        ref = ref.reshape((ref_dim[0]*60, ref_dim[1]*60))
    if len(calib) != 6:  # a bug in curve_fit - passed numpy array on last iteration instead of a tuple
        calib = calib[0].tolist()

    fov, pos_x, pos_y, pos_z, phi, theta = calib
    assert(pos_z > 0)  # camera has to be above the ground
    t0 = time.time()

    T = np.array([pos_x, pos_y, pos_z])
    R = Rotation.from_euler('zx', [phi, theta], degrees=True).as_matrix()
    # print(T, "\n", R)

    tan_z = np.tan(fov/2 * (np.pi/180))
    tan_x = tan_z * width / height
    x, z = np.meshgrid(np.linspace(-tan_x, tan_x, width), np.linspace(tan_z, -tan_z, height))
    ray = np.stack([x, np.ones((height, width)), z], axis=2).reshape(-1, 3)
    ray = np.matmul(ray, R)

    idx_r = np.nonzero(ray[:, 2] < 0)[0]  # only rays pointing downwards can intersect with ground plane
    g_xy = ray[idx_r, :2] * pos_z / (-ray[idx_r, 2][:, None]) + np.array([pos_x, pos_y])

    ri = ref.shape[0] * (ref_origin[0] - g_xy[:, 1]) / ref_dim[0]
    ci = ref.shape[1] * (ref_origin[1] + g_xy[:, 0]) / ref_dim[1]
    idx_rc = np.nonzero((ri >= 0) & (ri < ref.shape[0]) & (ci >= 0) & (ci < ref.shape[1]))[0]

    ri, ci = ri[idx_rc].astype(np.int), ci[idx_rc].astype(np.int)
    v = np.zeros(g_xy.shape[0])
    v[idx_rc] = ref[ri, ci]

    img = np.zeros((height * width), dtype=np.uint8)
    img[idx_r] = v
    img = img.reshape((height, width))
    # print("Rendering time:", time.time()-t0, "sec")

    if plot:
        plt.figure("Segmentation Internals", (12, 12))
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Segmentation Internals")

        p = ray[::1000, :] + T
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="g", s=5, label="rays")
        ax.scatter(g_xy[::1000, 0], g_xy[::1000, 1], np.zeros_like(g_xy[::1000, 0]), c="r", s=5, label="intersections")
        ax.scatter(pos_x, pos_y, pos_z, c="b", s=5, label="origin")

        basis(ax, T, R, length=1)

        field(ax, 400, 400)
        ax.set_xlabel("x, feet")
        ax.set_ylabel("y, feet")
        ax.set_zlabel("z, feet")
        plt.legend()
        plt.tight_layout()
        axis_equal_3d(ax)

        plt.figure("Segmentation Rendering")
        plt.imshow(img)
        plt.colorbar()

    return img.ravel()


if __name__ == "__main__":
    reference_filename = "../baseball_field/baseball_field_segmentation_5px_per_in"  # assumes png

    cache_reference(reference_filename, plot=True)
    ref = load_reference(reference_filename, plot=True)

    # vertical_fov (deg), position (x', y', z'), view_dir_phi (deg), view_dir_theta (deg)
    calib_true = (5, 400, 400, 50, 180+45, 5)
    ground_truth = render(ref, *calib_true, plot=True).reshape((height, width))

    calib_guess = (4.9, 410, 395, 45, 180+46, 4.9)
    initial_guess = render(ref, *calib_guess).reshape((height, width))

    # plt.figure("Ground Truth")
    # plt.imshow(ground_truth)
    # plt.colorbar()

    plt.figure("Initial Guess")
    plt.imshow(initial_guess)
    plt.colorbar()

    # diag = [1, 1/10, 1/10, 1/10, 1/10, 1]
    # diag = [1, 100, 100, 10, 100, 1]
    diag = [1, 1, 1, 1, 1, 1]

    popt, pcov = curve_fit(render, ref.ravel(), ground_truth.ravel(), calib_guess, epsfcn=0.01, factor=100, diag=diag)
    # popt = [4.97631985, 410.22153594, 394.98170901, 49.42902477, 226.06108039, 4.9159036]
    print("guess:", calib_guess)
    print("opt:", popt)
    print("true:", calib_true)
    optimized = render(ref, *popt).reshape((height, width))

    plt.figure("Overlay Before")
    plt.imshow(np.abs(ground_truth + initial_guess))
    plt.colorbar()

    plt.figure("Overlay After")
    plt.imshow(np.abs(ground_truth + optimized))
    plt.colorbar()

    plt.show()
