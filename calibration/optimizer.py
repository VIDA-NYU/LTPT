from old_optimizer import *
# tf.debugging.set_log_device_placement(True)
# from tensorflow_graphics.image.transformer import sample
from tensorflow_addons import image as tfa_image
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import timeit


def build_scaled_vars(calib, scales):
    print(calib, "\n", scales)

    fov, pos_x, pos_y, pos_z, phi, theta = calib
    s_fov, s_pos_x, s_pos_y, s_pos_z, s_phi, s_theta = scales

    fov, phi, theta = tf.Variable([fov / s_fov], dtype=tf.float32), \
                      tf.Variable([phi / s_phi], dtype=tf.float32), \
                      tf.Variable([theta / s_theta], dtype=tf.float32)

    pos_x, pos_y, pos_z = tf.Variable([pos_x / s_pos_x], dtype=tf.float32), \
                          tf.Variable([pos_y / s_pos_y], dtype=tf.float32), \
                          tf.Variable([pos_z / s_pos_z], dtype=tf.float32)

    s_fov, s_phi, s_theta = tf.constant(s_fov, dtype=tf.float32), \
                            tf.constant(s_phi, dtype=tf.float32), \
                            tf.constant(s_theta, dtype=tf.float32)

    s_pos_x, s_pos_y, s_pos_z = tf.constant(s_pos_x, dtype=tf.float32), \
                                tf.constant(s_pos_y, dtype=tf.float32), \
                                tf.constant(s_pos_z, dtype=tf.float32)

    vars = fov, pos_x, pos_y, pos_z, phi, theta
    scales = s_fov, s_pos_x, s_pos_y, s_pos_z, s_phi, s_theta

    return vars, scales


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

def trace_rays(tf_rays, vars, scales, ref, ref_shape=(25800, 25800), ref_dim=(430, 430), ref_origin=(355, 75)):
    t0 = time.time()
    fov, pos_x, pos_y, pos_z, phi, theta = vars
    s_fov, s_pos_x, s_pos_y, s_pos_z, s_phi, s_theta = scales

    to_rad = tf.constant(np.pi / 180.)
    ref_origin = tf.constant(ref_origin, dtype=tf.float32)
    ref_shape = tf.constant(ref_shape, dtype=tf.float32)
    ref_dim = tf.constant(ref_dim, dtype=tf.float32)

    rot_z = rotation_matrix_3d.from_axis_angle(tf.constant([0., 0., 1.]), phi * s_phi * to_rad)
    rot_x = rotation_matrix_3d.from_axis_angle(tf.constant([1., 0., 0.]), theta * s_theta * to_rad)
    rot = tf.matmul(rot_x, rot_z)

    tan_z = tf.tan(fov * s_fov * to_rad / 2.)
    sc = tf.concat([tan_z, tf.constant([1.]), tan_z], axis=0)
    tf_rays = tf.multiply(tf_rays, sc[None, :])
    tf_rays = tf.matmul(tf_rays, rot)
    # clamp rays pointing to the sky
    tf_rays = tf.minimum(tf_rays, tf.constant([1000., 1000., -0.001])[None, :])

    dist = tf.divide(pos_z * s_pos_z, -tf_rays[:, 2])
    xy = tf.multiply(tf_rays[:, :2], dist[:, None])
    xy = tf.add(xy, tf.concat([pos_x * s_pos_x, s_pos_y * pos_y], axis=0)[None, :])

    ri = tf.divide(tf.multiply(ref_shape[0], tf.subtract(ref_origin[0], xy[:, 1])), ref_dim[0])
    ci = tf.divide(tf.multiply(ref_shape[1], tf.add(ref_origin[1], xy[:, 0])), ref_dim[1])

    ri = tf.maximum(0., tf.minimum(ri, ref_shape[0]-1.))
    ci = tf.maximum(0., tf.minimum(ci, ref_shape[1]-1.))

    cri = tf.stack([ci, ri], axis=1)
    v = tfa_image.resampler(ref[None, :, :, None], cri[None, :, :])
    print("Tracing time:", time.time() - t0, "sec")
    # return ri
    return v[0, :, 0]


@tf.function
def graph_trace_rays(tf_rays, vars, scales, ref):
    return trace_rays(tf_rays, vars, scales, ref)


def perf_test(ref, *calib, height=1080, width=1920, mask=None, plot=True, title="Unknown"):
    @tf.function
    def graph_trace_rays(tf_rays, vars, scales, ref):
        with tf.GradientTape(persistent=True) as tape:
            v = trace_rays(tf_rays, vars, scales, ref)
        return v

    vars, scales = build_scaled_vars(calib, [1, 1, 1, 1, 1, 1])

    np_rays, r_idx = prep_rays(height, width, mask, plot=False)
    tf_rays = tf.convert_to_tensor(value=np_rays, dtype=tf.float32)

    print("Eager time 1:", timeit.timeit(lambda: trace_rays(tf_rays, vars, scales, ref), number=1))
    print("Eager time 10:", timeit.timeit(lambda: trace_rays(tf_rays, vars, scales, ref), number=10))
    print("Graph time 1:", timeit.timeit(lambda: graph_trace_rays(tf_rays, vars, scales, ref), number=1))
    print("Graph time 10:", timeit.timeit(lambda: graph_trace_rays(tf_rays, vars, scales, ref), number=10))


def opt_render(ref, *calib, height=1080, width=1920, mask=None, plot=True, title="Unknown"):
    vars, scales = build_scaled_vars(calib, [1, 1, 1, 1, 1, 1])

    np_rays, r_idx = prep_rays(height, width, mask, plot=False)
    tf_rays = tf.convert_to_tensor(value=np_rays, dtype=tf.float32)

    values = trace_rays(tf_rays, vars, scales, ref)

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

    # vertical_fov (deg), position (x', y', z'), view_dir_phi (deg), view_dir_theta (deg)
    vars, scales = build_scaled_vars(calib_guess, [1, 100, 100, 10, 10, 1])
    # vars, scales = build_scaled_vars(calib_guess, [1, 1, 1, 1, 1, 1])

    np_rays, r_idx = prep_rays(height, width, mask, plot=True)
    tf_rays = tf.constant(tf.convert_to_tensor(value=np_rays, dtype=tf.float32))

    def residuals(*vars):
        print(vars)
        if len(vars) != 6:  # tensorflow passed vars wrapped into extra tuple
            vars = vars[0]

        t0 = time.time()
        current = trace_rays(tf_rays, vars, scales, ref)
        print(current)
        print("Iteration time:", time.time() - t0, "sec")

        return tf.cast(current - ground_truth[1], tf.float32)

    # Optimization.
    # _, opt_vars = levenberg_marquardt.minimize(residuals, vars, 3)#, experimental_relax_shapes=True)

    gt = tf.constant(tf.cast(ground_truth[1], tf.float32))

    with tf.GradientTape(persistent=True) as tape:
        v = trace_rays(tf_rays, vars, scales, ref)
        # v = vars[0] * 2
        # print(v)

    print(v)
    print(vars[1])
    print(tape.gradient(v, vars[1]))
    # print(tape.gradient(v, vars[0]).numpy())

    # <tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([4.1], dtype=float32)>
    # tf.Tensor([273386.72], shape=(1,), dtype=float32)

    # @tf.function
    def loss():
        tf.print("vars:", vars)

        t0 = time.time()
        current = graph_trace_rays(tf_rays, vars, scales, ref)
        l = tf.keras.losses.mean_squared_error(current, gt)
        print("Iteration time:", time.time() - t0, "sec")
        return l

    cost = lambda: tf.keras.losses.mean_squared_error(tf.cast(trace_rays(tf_rays, vars, scales, ref), tf.int32), gt)

    opt = tf.keras.optimizers.SGD(learning_rate=0.02)
    for i in range(10):
        step_count = opt.minimize(loss, vars).numpy()
        print(step_count)
    opt_vars = vars

    # # @tf.function
    # def gradients(tf_rays, vars, scales, ref, gt):
    #     t0 = time.time()
    #     with tf.GradientTape(persistent=False) as tape:
    #         current = graph_trace_rays(tf_rays, vars, scales, ref)
    #         L = tf.keras.losses.mean_squared_error(current, gt)
    #
    #     tf.print("Tape time:", time.time() - t0, "sec")
    #     t1 = time.time()
    #     gradients = tape.gradient(L, vars)
    #     tf.print("Gradients time:", time.time() - t1, "sec")
    #     return gradients
    #
    # # @tf.function
    # def train_step(tf_rays, vars, scales, ref, gt, opt):
    #     t0 = time.time()
    #     tf.print("vars:", vars)
    #
    #     # with tf.GradientTape() as tape:
    #     #     current = trace_rays(tf_rays, vars, scales, ref)
    #     #     L = tf.keras.losses.mean_squared_error(current, gt)
    #     #
    #     # gradients = tape.gradient(L, vars)
    #     g = gradients(tf_rays, vars, scales, ref, gt)
    #     t1 = time.time()
    #     opt.apply_gradients(zip(g, vars))
    #     print("Apply Gradients time:", time.time() - t1, "sec")
    #
    #     print("Iteration time:", time.time() - t0, "sec")
    #
    # opt = tf.keras.optimizers.SGD(learning_rate=0.02)
    # print("\nOptimizing")
    #
    # for i in range(10):
    #     train_step(tf_rays, vars, scales, ref, gt, opt)
    #     print("step", i)

    tf.print("opt_vars:", [var * scale for var, scale in zip(vars, scales)])

    if plot:
        plt.figure("Overlay Before")
        plt.imshow(np.abs(ground_truth[0] + initial_guess[0]))
        plt.colorbar()

        optimal = trace_rays(tf_rays, vars, scales, ref)

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

    # perf_test(ref, *calib_true, mask=mask, plot=True, title="Ground Truth")
    # exit()

    ground_truth = opt_render(ref, *calib_true, mask=mask, plot=True, title="Ground Truth")

    # calib_guess = (4.5, 450, 390, 65, 180 + 47, 4.9)
    calib_guess = (4.9, 410, 395, 45, 180 + 46, 4.9)
    calib_opt = cam_calib(ref, ground_truth, calib_guess, mask=mask, plot=True)

    plt.show()
