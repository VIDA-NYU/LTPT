import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_addons import image as tfa_image


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
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


class Calibrator:
    def __init__(self, ref, mask, defaults=None, scales=None, plot=False):
        # self.optimizer = tf.keras.optimizers.SGD()
        self.optimizer = tf.keras.optimizers.Adam()

        self.prep_ref(ref, plot=plot)
        self.prep_rays(mask, plot=plot)

        self.build_scaled_vars(defaults or [0] * 7, scales or [100, 100, 10, 100, 50, 100, 5])
        self.reset()

    def prep_ref(self, ref, ref_dim=(430, 430), ref_origin=(355, 75), plot=False):
        # assert(ref.shape == (5160, 5160))
        print("Reference:", ref.shape)
        ref[[0, -1], :] = 0
        ref[:, [0, -1]] = 0

        self.ref = tf.convert_to_tensor(value=ref, dtype=tf.float32)
        self.ref_shape = tf.constant(ref.shape, dtype=tf.float32)

        self.ref_dim = tf.constant(ref_dim, dtype=tf.float32)
        self.ref_origin = tf.constant(ref_origin, dtype=tf.float32)

        if plot:
            plt.figure("Reference", (16, 9))
            plt.imshow(ref)
            plt.colorbar()
            plt.tight_layout()

    def prep_rays(self, mask, plot=False):
        print("Resolution: ", mask.shape)
        h, w = mask.shape
        x, z = np.meshgrid(np.linspace(-w / h, w / h, w), np.linspace(1, -1, h))
        rays = np.stack([x, np.ones(mask.shape), z], axis=2).reshape(-1, 3)

        r_idx = np.nonzero(mask.ravel())[0]
        rays = rays[r_idx, :]

        if plot:
            plt.figure("Rays", (12, 12))
            ax = plt.subplot(111, projection='3d', proj_type='ortho')
            ax.set_title("Rays")

            p = rays[np.random.randint(0, rays.shape[0], 3000), :]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="g", s=5, label="Rays")
            basis(ax, np.zeros(3), np.eye(3), length=1)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.legend()
            plt.tight_layout()
            axis_equal_3d(ax)

        self.width = tf.constant(w, dtype=tf.int32)
        self.height = tf.constant(h, dtype=tf.int32)
        self.rays = tf.convert_to_tensor(value=rays, dtype=tf.float32)
        self.r_idx = tf.convert_to_tensor(value=r_idx, dtype=tf.int32)

    def build_scaled_vars(self, defaults, scales):
        print("Defaults: ", defaults)
        print("Scales: ", scales)

        self.defaults, self.scales = defaults, scales
        self.to_rad = tf.constant(np.pi / 180.)

        self.all_scales = [tf.constant(scale, dtype=tf.float32) for scale in scales]
        self.s_pos_x, self.s_pos_y, self.s_pos_z, self.s_phi, self.s_theta, self.s_dist, self.s_fov = self.all_scales

        self.all_vars = [tf.Variable([default / scale], dtype=tf.float32) for default, scale in zip(defaults, scales)]
        self.pos_x, self.pos_y, self.pos_z, self.phi, self.theta, self.dist, self.fov = self.all_vars

        self.loc_vars = [self.pos_x, self.pos_y, self.pos_z, self.phi, self.theta]
        self.pos_vars = [self.pos_x, self.pos_y, self.pos_z]
        self.rot_vars = [self.phi, self.theta]
        self.cam_vars = [self.theta, self.dist, self.fov]

    def reset(self):
        self.losses, self.configs = [], []

        for var, default, scale in zip(self.all_vars, self.defaults, self.scales):
            var.assign([default / scale])
        tf.print("Default vars:", self.all_vars)

    def trace_rays(self, inputs):
        pos_x, pos_y, pos_z, phi, theta, dist, fov = inputs

        pos_x = pos_x + self.pos_x * self.s_pos_x
        pos_y = pos_y + self.pos_y * self.s_pos_y
        pos_z = pos_z + self.pos_z * self.s_pos_z
        dist = dist + self.dist * self.s_dist

        phi = (phi + self.phi * self.s_phi) * self.to_rad
        theta = (theta + self.theta * self.s_theta) * self.to_rad
        fov = (fov + self.fov * self.s_fov) * self.to_rad

        # angle > 0 means clockwise in rotation_matrix_3d
        rot_z = rotation_matrix_3d.from_axis_angle(tf.constant([0., 0., 1.]), phi)
        rot_x = rotation_matrix_3d.from_axis_angle(tf.constant([1., 0., 0.]), theta)
        rot = tf.matmul(rot_x, rot_z)

        tan_z = tf.tan(fov / 2.)
        sc = tf.concat([tan_z, tf.constant([1.]), tan_z], axis=0)
        rays = tf.multiply(self.rays, sc[None, :])
        rays = tf.matmul(rays, rot)
        rays = tf.minimum(rays, tf.constant([1000., 1000., -0.001])[None, :])  # clamp rays pointing to the sky

        axis = tf.matmul(tf.constant([0., 1., 0.])[None, :], rot)[0, :]
        height = pos_z - dist * axis[2]
        offset = tf.concat([pos_x, pos_y], axis=0) - dist * axis[:2]
        xy = tf.multiply(rays[:, :2], tf.divide(height, -rays[:, 2])[:, None])
        xy = tf.add(xy, offset[None, :])

        ri = tf.divide(tf.multiply(self.ref_shape[0], tf.subtract(self.ref_origin[0], xy[:, 1])), self.ref_dim[0])
        ci = tf.divide(tf.multiply(self.ref_shape[1], tf.add(self.ref_origin[1], xy[:, 0])), self.ref_dim[1])

        ri = tf.maximum(0., tf.minimum(ri, self.ref_shape[0] - 1.))
        ci = tf.maximum(0., tf.minimum(ci, self.ref_shape[1] - 1.))

        cri = tf.stack([ci, ri], axis=1)
        values = tfa_image.resampler(self.ref[None, :, :, None], cri[None, :, :])

        return values[0, :, 0]

    def make_image(self, values, plot=False, title="Image"):
        shape = (self.height.numpy(), self.width.numpy())
        img = np.zeros(shape[0] * shape[1])
        img[self.r_idx.numpy()] = values.numpy()
        img = img.reshape(shape)

        if plot:
            plt.figure(title, (16, 9))
            plt.imshow(img)
            plt.colorbar()
            plt.tight_layout()

        return img

    def overlay(self, img_1, img_2, title):
        plt.figure("Overlay " + title, (16, 9))
        plt.gcf().clear()
        plt.imshow(img_1 + img_2)
        plt.colorbar()
        plt.tight_layout()

    def plot_loss(self):
        plt.figure("Loss")
        plt.gcf().clear()
        plt.plot(np.arange(len(self.losses)), self.losses)
        plt.tight_layout()

    def __call__(self, inputs):
        return self.trace_rays(inputs)

    def update(self, guess, gt, vars=None):
        i, config = self.optimizer.iterations, self.get_config(guess)
        self.configs.append(config)
        print("Config %d:" % i, config)

        with tf.GradientTape() as tape:
            loss = tf.keras.losses.mean_squared_error(self(guess), gt)

        self.losses.append(loss.numpy())
        vars = vars or self.all_vars
        grads = tape.gradient(loss, vars)

        self.optimizer.apply_gradients(zip(grads, vars))

    def get_config(self, base):
        adjustments = [float((var * scale).numpy()) for var, scale in zip(self.all_vars, self.all_scales)]
        return tuple([b + a for b, a in zip(base, adjustments)])

    def set_config(self, config, base):
        adjustments = [c - b for c, b in zip (config, base)]

        for var, adj, scale in zip(self.all_vars, adjustments, self.scales):
            var.assign([adj / scale])
        tf.print("Chosen vars:", self.all_vars)

    def optimize(self, guess, gt, vars=None, epoch=50, rate=0.01, decay=1., plot=False, skip=5, last=True):
        if plot:
            gt_img = self.make_image(gt, plot=plot, title="Ground Truth")
            guess_img = self.make_image(self(guess), plot=plot, title="Initial Guess")
            self.overlay(gt_img, guess_img, "Before")

        print("\nOptimizing...")
        tf.print("Init vars:", self.all_vars)
        t0 = time.time()
        vars = vars or self.all_vars

        for i in range(epoch):
            self.optimizer.learning_rate = rate * np.exp(-decay * i / epoch)

            ti = time.time()
            self.update(guess, gt, vars=vars)
            time.sleep(0.01)
            # print("Iteration %d:" % i, time.time() - ti, "sec", "\n")

            if plot and i % skip == 0:
                self.overlay(gt_img, self.make_image(self(guess)), "Progress")
                plt.pause(0.01)
                self.plot_loss()
                plt.pause(0.01)

        tf.print("Last vars:", self.all_vars)
        print("Optimization time:", time.time() - t0, "sec")

        best = int(np.argmin(self.losses))
        self.best_config = self.configs[best]
        print("Best config %d:" % best, self.best_config)
        print("Min loss so far:", self.losses[best])
        self.set_config(self.best_config, guess)

        if plot and last:
            self.reset()  # To plot with opt_config instead of guess + vars
            opt_img = self.make_image(self(self.best_config), plot=plot, title="Last Iteration")
            self.overlay(gt_img, opt_img, "After")
            self.plot_loss()

        return self.best_config

    def calibrate(self, guess, gt, **kw):
        t0 = time.time()
        self.optimize(guess, gt, vars=self.all_vars, last=False, epoch=50, rate=0.02, decay=2, **kw)
        self.optimize(guess, gt, vars=self.loc_vars, last=False, epoch=50, rate=0.01, decay=2, **kw)
        self.optimize(guess, gt, vars=self.rot_vars, last=False, epoch=30, rate=0.01, decay=1, **kw)
        self.optimize(guess, gt, vars=self.cam_vars, last=False, epoch=20, rate=0.005, decay=1, **kw)
        self.optimize(guess, gt, vars=self.all_vars, last=True, epoch=20, rate=0.001, decay=1, **kw)
        print("Total optimization time:", time.time() - t0, "sec")
        return self.best_config


def load_reference(filename, plot=False):
    ref = np.load(filename + ".npy")
    ref = ref[2::5, 2::5]  # make it 1px/in (faster optimization)
    # print("reference:", ref.shape, ref.dtype)

    if plot:
        plt.figure("Gray Reference", (16, 9))
        plt.imshow(ref)
        plt.colorbar()
        plt.tight_layout()

    return ref


def gen_mask(w, h, plot=False):
    mask = np.ones((h, w), dtype=np.bool)
    mask[750:, 1500:] = 0

    if plot:
        plt.figure("Mask", (16, 9))
        plt.imshow(mask)
        plt.tight_layout()

    return mask


def validate():
    reference_filename = "../baseball_field/baseball_field_segmentation_5px_per_in"

    ref = load_reference(reference_filename, plot=False)
    mask = gen_mask(1920, 1080, plot=False)

    cal = Calibrator(ref, mask, plot=True)

    # Behind Pitcher
    calib_true = (45, 45, 5, 180 + 45, 5, 400, 5)
    calib_guess = (55, 50, 7, 180 + 46, 6, 350, 4.5)

    # Pitcher Side (Right Handed)
    # calib_true = (45, 45, 10, 90 + 55, 7, 200, 10)
    # calib_guess = (75, 30, 6, 90 + 65, 6, 250, 8)

    # Batter Side (Right Handed)
    # calib_true = (0, 0, 5, -55, 4, 100, 15)
    # calib_guess = (5, 10, 3, -40, 3, 122, 12)

    # Behind Batter
    # calib_true = (40, 40, 0, 45, 20, 300, 25)
    # calib_guess = (50, 60, 10, 50, 15, 330, 22)

    calib_opt = cal.calibrate(calib_guess, cal(calib_true), plot=True)
    print("\nOptimal calibration:", calib_opt)


def test():
    pass


if __name__ == "__main__":
    validate()
    # test()

    plt.show()
