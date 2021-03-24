from old_optimizer import *
# tf.debugging.set_log_device_placement(True)
# from tensorflow_graphics.image.transformer import sample
from tensorflow_addons import image as tfa_image


def prep_rays(height=1080, width=1920, mask=None, plot=True):
    aspect = width / height
    x, z = np.meshgrid(np.linspace(-aspect, aspect, width), np.linspace(1, -1, height))
    rays = np.stack([x, np.ones((height, width)), z], axis=2).reshape(-1, 3)

    if mask is not None:
        r_idx = np.nonzero(mask.ravel())[0]
        rays = rays[r_idx, :]
    else:
        r_idx = np.arange(width * height)

    if plot:
        plt.figure("Rays", (12, 12))
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Rays")

        p = rays[np.random.randint(0, rays.shape[0], 1000), :]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="g", s=5, label="rays")
        basis(ax, np.zeros(3), np.eye(3), length=1)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.legend()
        plt.tight_layout()
        axis_equal_3d(ax)

    return rays, r_idx


def trace_rays(tf_rays, vars, ref, ref_dim=(430, 430), ref_origin=(355, 75)):
    fov, pos_x, pos_y, pos_z, phi, theta = vars
    assert (pos_z > 0)  # camera has to be above the ground
    # print(ref.shape, ref)

    to_rad = tf.constant(np.pi / 180.)
    ref_origin = tf.constant(ref_origin, dtype=tf.float32)
    ref_dim = tf.constant(ref_dim, dtype=tf.float32)
    # trans = tf.concat([pos_x, pos_y, pos_z], axis=0)
    rot_z = rotation_matrix_3d.from_axis_angle(tf.constant([0, 0, 1.]), phi * to_rad)
    rot_x = rotation_matrix_3d.from_axis_angle(tf.constant([1., 0, 0]), theta * to_rad)
    rot = tf.matmul(rot_x, rot_z)

    tan_z = tf.tan(fov * to_rad / 2.)
    sc = tf.concat([tan_z, tf.constant([1.]), tan_z], axis=0)
    # print(sc)
    # tf_rays = tf.convert_to_tensor(value=np_rays, dtype=tf.float32)
    tf_rays = tf.multiply(tf_rays, sc[None, :])
    tf_rays = tf.matmul(tf_rays, rot)

    # r = tf.reduce_sum(tf_rays, axis=1)
    # print(r)
    # return r

    dist = tf.divide(pos_z, -tf_rays[:, 2])
    xy = tf.multiply(tf_rays[:, :2], dist[:, None])
    xy = tf.add(xy, tf.concat([pos_x, pos_y], axis=0)[None, :])

    # r = tf.reduce_sum(xy, axis=1)
    # print(r)
    # return r

    ri = tf.divide(tf.multiply(ref.shape[0], tf.subtract(ref_origin[0], xy[:, 1])), ref_dim[0])
    ci = tf.divide(tf.multiply(ref.shape[1], tf.add(ref_origin[1], xy[:, 0])), ref_dim[1])

    ri = tf.maximum(0, tf.minimum(ri, ref.shape[0]-1))
    ci = tf.maximum(0, tf.minimum(ci, ref.shape[1]-1))

    cri = tf.stack([ci, ri], axis=1)
    v = tfa_image.resampler(ref[None, :, :, None], cri[None, :, :])
    # # ri = tf.floor(ri)
    # # print(v)
    return v[0, :, 0]

    ri = tf.cast(tf.maximum(0, tf.minimum(ri, ref.shape[0]-1)), tf.int32)
    ci = tf.cast(tf.maximum(0, tf.minimum(ci, ref.shape[1]-1)), tf.int32)

    rci = tf.stack([ri, ci], axis=1)
    return tf.gather_nd(ref, rci)

    # return tf.cast(ri, tf.float32)


def opt_render(ref, *calib, height=1080, width=1920, mask=None, plot=True, title="Unknown"):
    vars = build_vars(calib)

    np_rays, r_idx = prep_rays(height, width, mask, plot=False)
    tf_rays = tf.convert_to_tensor(value=np_rays, dtype=tf.float32)

    values = trace_rays(tf_rays, vars, ref)

    if plot:
        img = np.zeros(height*width)
        img[r_idx] = values.numpy()
        img = img.reshape((height, width))

        plt.figure("Rendered " + title)
        plt.imshow(img)
        plt.colorbar()
    else:
        img = None

    return img, values, r_idx


def cam_calib(ref, ground_truth, calib_guess, height=1080, width=1920, mask=None, plot=False):
    initial_guess = opt_render(ref, *calib_guess, mask=mask, plot=True, title="Initial Guess")
    # print(ground_truth, "\n\n", initial_guess)

    vars = build_vars(calib_guess)

    np_rays, r_idx = prep_rays(height, width, mask, plot=True)
    tf_rays = tf.constant(tf.convert_to_tensor(value=np_rays, dtype=tf.float32))

    def residuals(*vars):
        print(vars)
        if len(vars) != 6:  # tensorflow passed vars wrapped into extra tuple
            vars = vars[0]

        t0 = time.time()
        current = trace_rays(tf_rays, vars, ref)
        print(current)
        print("Iteration time:", time.time() - t0, "sec")

        return tf.cast(current - ground_truth[1], tf.float32)

    # Optimization.
    # _, opt_vars = levenberg_marquardt.minimize(residuals, vars, 3)#, experimental_relax_shapes=True)

    gt = tf.constant(tf.cast(ground_truth[1], tf.float32))

    with tf.GradientTape(persistent=True) as tape:
        v = trace_rays(tf_rays, vars, ref)
        # v = vars[0] * 2
        # print(v)

    print(v)
    print(vars[0])
    print(tape.gradient(v, vars[0]))
    # print(tape.gradient(v, vars[0]).numpy())

    def loss():
        tf.print("vars:", vars)

        t0 = time.time()
        current = tf.cast(trace_rays(tf_rays, vars, ref), tf.float32)
        print("Iteration time:", time.time() - t0, "sec")

        return tf.keras.losses.mean_squared_error(current, gt)

    cost = lambda: tf.keras.losses.mean_squared_error(tf.cast(trace_rays(tf_rays, vars, ref), tf.int32), gt)

    opt = tf.keras.optimizers.SGD(learning_rate=0.3)
    for i in range(30):
        step_count = opt.minimize(loss, vars).numpy()
        print(step_count)
    opt_vars = vars

    tf.print("opt_vars:", opt_vars)

    if plot:
        plt.figure("Overlay Before")
        plt.imshow(np.abs(ground_truth[0] + initial_guess[0]))
        plt.colorbar()

        optimal = trace_rays(tf_rays, opt_vars, ref)

        img = np.zeros(height*width)
        img[r_idx] = optimal.numpy()
        img = img.reshape((height, width))

        plt.figure("Overlay After")
        plt.imshow(np.abs(ground_truth[0] + img))
        plt.colorbar()


if __name__ == "__main__":
    reference_filename = "../baseball_field/baseball_field_segmentation_5px_per_in"  # assumes png
    ref = load_reference(reference_filename, plot=True)
    ref = tf.constant(tf.convert_to_tensor(value=ref, dtype=tf.float32))
    w, h = 1920, 1080

    mask = np.ones((h, w), dtype=np.bool)
    mask[750:, 1500:] = 0

    plt.figure("Mask")
    plt.imshow(mask)

    # vertical_fov (deg), position (x', y', z'), view_dir_phi (deg), view_dir_theta (deg)
    calib_true = (5, 400, 400, 50, 180 + 45, 5)
    ground_truth = opt_render(ref, *calib_true, mask=mask, plot=True, title="Ground Truth")

    calib_guess = (4.9, 410, 395, 45, 180 + 46, 4.9)
    calib_opt = cam_calib(ref, ground_truth, calib_guess, mask=mask, plot=True)

    plt.show()
