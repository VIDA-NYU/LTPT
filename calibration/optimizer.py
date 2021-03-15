import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit
import tensorflow as tf
from tensorflow_graphics.rendering.camera import perspective
from tensorflow_graphics.geometry.transformation import quaternion, rotation_matrix_3d
from tensorflow_graphics.math.optimizer import levenberg_marquardt

# tf.enable_eager_execution()


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

def tf_render(ref, *vars, im_width=1920*2, im_height=1080*2, ref_dim=(430, 430), ref_origin=(355, 75), plot=False):
    if len(vars) != 6:  # tensorflow passed vars wrapped into extra tuple
        vars = vars[0]

    print(vars)

    fov, pos_x, pos_y, pos_z, phi, theta = vars
    assert(pos_z > 0)  # camera has to be above the ground
    t0 = time.time()

    width, height, to_rad = tf.constant(im_width), tf.constant(im_height), tf.constant(np.pi / 180.)
    ref_origin, ref_dim = tf.constant(ref_origin, dtype=tf.float32), tf.constant(ref_dim, dtype=tf.float32)

    # fov, phi, theta = tf.Variable([fov], dtype=tf.float32),\
    #                   tf.Variable([phi], dtype=tf.float32),\
    #                   tf.Variable([theta], dtype=tf.float32)
    #
    # pos_x, pos_y, pos_z = tf.Variable([pos_x], dtype=tf.float32),\
    #                       tf.Variable([pos_y], dtype=tf.float32),\
    #                       tf.Variable([pos_z], dtype=tf.float32)

    # print(width, height, to_rad)
    # print(fov, phi, theta)
    # print(pos_x, pos_y, pos_z)

    trans = tf.concat([pos_x, pos_y, pos_z], axis=0)
    rot_z = rotation_matrix_3d.from_axis_angle(tf.constant([0, 0, 1.]), phi * to_rad)
    rot_x = rotation_matrix_3d.from_axis_angle(tf.constant([1., 0, 0]), theta * to_rad)
    rot = tf.matmul(rot_x, rot_z)
    # print(trans, "\n", rot)

    # T = np.array([pos_x, pos_y, pos_z])
    # R = Rotation.from_euler('zx', [phi, theta], degrees=True).as_matrix()
    # print(T, "\n", R)

    tan_z = tf.tan(fov * to_rad / 2.)
    tan_x = tan_z * tf.cast(width, tf.float32) / tf.cast(height, tf.float32)
    x, z = tf.meshgrid(tf.linspace(-tan_x, tan_x, width), tf.linspace(tan_z, -tan_z, height))
    ray = tf.stack([x, tf.ones((height, width)), z], axis=2)
    ray = tf.reshape(ray, [-1, 3])
    ray = tf.matmul(ray, rot)
    # print(tan_z, tan_x, "\n")
    # print(ray)

    idx_r = tf.where(ray[:, 2] < 0)[:, 0]  # only rays pointing downwards can intersect with ground plane
    # print("idx_r:", idx_r)
    # print(idx_r)

    dist = tf.divide(pos_z, -tf.gather(ray[:, 2], idx_r))
    # print(dist)
    xy = tf.gather(ray[:, :2], idx_r[:, None])[:, 0, :]
    # print(xy)

    g_xy = tf.multiply(xy,  dist[:, None])
    g_xy = tf.add(g_xy, tf.concat([pos_x, pos_y], axis=0)[None, :])
    # print("g_xy:", g_xy)
    # g_xy = ray[idx_r, :2] * pos_z / (-ray[idx_r, 2][:, None]) + tf.concat([pos_x, pos_y], axis=0)

    # ref = tf.convert_to_tensor(value=ref)
    # print(ref.shape)

    ri = tf.divide(tf.multiply(ref.shape[0], tf.subtract(ref_origin[0], g_xy[:, 1])), ref_dim[0])
    ci = tf.divide(tf.multiply(ref.shape[1], tf.add(ref_origin[1], g_xy[:, 0])), ref_dim[1])
    # print(ri, ci)
    # ri = ref.shape[0] * (ref_origin[0] - g_xy[:, 1]) / ref_dim[0]
    # ci = ref.shape[1] * (ref_origin[1] + g_xy[:, 0]) / ref_dim[1]

    idx_rc = tf.where((ri >= 0) & (ri < ref.shape[0]) & (ci >= 0) & (ci < ref.shape[1]))[:, 0]
    # print("idx_rc:", idx_rc)

    ri = tf.cast(tf.gather(ri, idx_rc), tf.int32)
    ci = tf.cast(tf.gather(ci, idx_rc), tf.int32)

    rci = tf.stack([ri, ci], axis=1)
    # print("rci:", rci)
    v = tf.gather_nd(ref, rci)
    # print("v:", v)
    v2 = tf.scatter_nd(idx_rc[:, None], v, [g_xy.shape[0]])
    # print("v2:", v2)
    v3 = tf.scatter_nd(idx_r[:, None], v2, [ray.shape[0]])
    # print("v3:", v3)

    img = v3
    # img = tf.zeros((height * width), dtype=tf.uint8)
    # # img[idx_r] = v
    # img = tf.reshape(img, [height, width])

    print("Rendering time:", time.time()-t0, "sec")

    if plot:
        plt.figure("Segmentation Internals", (12, 12))
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Segmentation Internals")

        ray, g_xy = ray.numpy(), g_xy.numpy()
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
        plt.imshow(img.numpy().reshape((height, width)))
        plt.colorbar()

    return img


def build_vars(calib):
    print(calib)
    fov, pos_x, pos_y, pos_z, phi, theta = calib

    fov, phi, theta = tf.Variable([fov], dtype=tf.float32), \
                      tf.Variable([phi], dtype=tf.float32), \
                      tf.Variable([theta], dtype=tf.float32)

    pos_x, pos_y, pos_z = tf.Variable([pos_x], dtype=tf.float32), \
                          tf.Variable([pos_y], dtype=tf.float32), \
                          tf.Variable([pos_z], dtype=tf.float32)

    return fov, pos_x, pos_y, pos_z, phi, theta


def tf_optimize(ref, ground_truth, calib_guess):
    ref = tf.convert_to_tensor(value=ref)
    print("ref:", ref.shape)

    ground_truth = tf.reshape(tf.convert_to_tensor(value=ground_truth), (-1,))
    print("ref:", ground_truth.shape)

    vars = build_vars(calib_guess)

    def residuals(*vars):
        current = tf_render(ref, vars)
        return tf.cast(current - ground_truth, tf.float32)

    # Optimization.
    _, opt_vars = levenberg_marquardt.minimize(residuals, vars, 10)#, experimental_relax_shapes=True)

    print("opt_vars:", opt_vars)

    return opt_vars


# ref_dim and ref_origin in feet (x-right, y-down, top-left image origin)
#
def np_render(ref, *calib, ref_dim=(430, 430), ref_origin=(355, 75), plot=False):
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


def scipy_optimize(ref, ground_truth, calib_guess):
    # diag = [1, 1/10, 1/10, 1/10, 1/10, 1]
    # diag = [1, 100, 100, 10, 100, 1]
    diag = [1, 1, 1, 1, 1, 1]

    # popt, pcov = curve_fit(render, ref.ravel(), ground_truth.ravel(), calib_guess, epsfcn=0.01, factor=100, diag=diag)
    popt = [4.97631985, 410.22153594, 394.98170901, 49.42902477, 226.06108039, 4.9159036]
    print("guess:", calib_guess)
    print("opt:", popt)
    print("true:", calib_true)

    return popt


if __name__ == "__main__":
    reference_filename = "../baseball_field/baseball_field_segmentation_5px_per_in"  # assumes png

    # cache_reference(reference_filename, plot=True)

    ref = load_reference(reference_filename, plot=False)

    # vertical_fov (deg), position (x', y', z'), view_dir_phi (deg), view_dir_theta (deg)
    calib_true = (5, 400, 400, 50, 180+45, 5)

    # test = tf_render(tf.convert_to_tensor(value=ref), *build_vars(calib_true), plot=True).numpy().reshape((height, width))

    ground_truth = np_render(ref, *calib_true, plot=False).reshape((height, width))

    plt.figure("Ground Truth")
    plt.imshow(ground_truth)
    plt.colorbar()

    # plt.show()
    # exit()

    calib_guess = (4.9, 410, 395, 45, 180+46, 4.9)
    initial_guess = np_render(ref, *calib_guess).reshape((height, width))

    plt.figure("Initial Guess")
    plt.imshow(initial_guess)
    plt.colorbar()

    # calib_opt = scipy_optimize(ref, ground_truth, calib_guess)
    # optimized = np_render(ref, *calib_opt).reshape((height, width))

    calib_opt = tf_optimize(ref, ground_truth, calib_guess)
    optimized = tf_render(tf.convert_to_tensor(value=ref), *calib_opt).numpy().reshape((height, width))

    plt.figure("Overlay Before")
    plt.imshow(np.abs(ground_truth + initial_guess))
    plt.colorbar()

    plt.figure("Overlay After")
    plt.imshow(np.abs(ground_truth + optimized))
    plt.colorbar()

    plt.show()
