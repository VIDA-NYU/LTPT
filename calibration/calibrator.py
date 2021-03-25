import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_addons import image as tfa_image

# tf.set_default_graph()


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


class Calibrator:  # (keras.Model):
    def __init__(self, ref, mask, rate=0.02, defaults=None, scales=None, plot=False, **kwargs):
        # super(keras.Model, self).__init__(**kwargs)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=rate)

        self.prep_ref(ref, plot=plot)
        self.prep_rays(mask, plot=plot)

        self.build_scaled_vars(defaults or [0] * 6, scales or [1] * 6)

    def prep_ref(self, ref, ref_dim=(430, 430), ref_origin=(355, 75), plot=False):
        # assert(ref.shape == (5160, 5160))

        self.ref = tf.convert_to_tensor(value=ref, dtype=tf.float32)
        self.ref_shape = tf.constant(ref.shape, dtype=tf.float32)

        self.ref_dim = tf.constant(ref_dim, dtype=tf.float32)
        self.ref_origin = tf.constant(ref_origin, dtype=tf.float32)

    def prep_rays(self, mask, plot=False):
        print("resolution: ", mask.shape)
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
        print("defaults: ", defaults)
        print("scales: ", scales)

        self.to_rad = tf.constant(np.pi / 180.)

        self.all_scales = [tf.constant(scale, dtype=tf.float32) for scale in scales]
        self.s_fov, self.s_pos_x, self.s_pos_y, self.s_pos_z, self.s_phi, self.s_theta = self.all_scales

        self.all_vars = [tf.Variable([default / scale], dtype=tf.float32) for default, scale in zip(defaults, scales)]
        self.fov, self.pos_x, self.pos_y, self.pos_z, self.phi, self.theta = self.all_vars

        tf.print("init_vars:", self.all_vars)

    # @tf.function
    def trace_rays(self, inputs):
        t0 = tf.timestamp()
        # t1 = time.time()

        fov, pos_x, pos_y, pos_z, phi, theta = inputs

        fov = (fov + self.fov * self.s_fov) * self.to_rad
        phi = (phi + self.phi * self.s_phi) * self.to_rad
        theta = (theta + self.theta * self.s_theta) * self.to_rad

        pos_x = pos_x + self.pos_x * self.s_pos_x
        pos_y = pos_y + self.pos_y * self.s_pos_y
        pos_z = pos_z + self.pos_z * self.s_pos_z

        rot_z = rotation_matrix_3d.from_axis_angle(tf.constant([0., 0., 1.]), phi)
        rot_x = rotation_matrix_3d.from_axis_angle(tf.constant([1., 0., 0.]), theta)
        rot = tf.matmul(rot_x, rot_z)

        tan_z = tf.tan(fov / 2.)
        sc = tf.concat([tan_z, tf.constant([1.]), tan_z], axis=0)
        rays = tf.multiply(self.rays, sc[None, :])
        rays = tf.matmul(rays, rot)
        # clamp rays pointing to the sky
        rays = tf.minimum(rays, tf.constant([1000., 1000., -0.001])[None, :])

        dist = tf.divide(pos_z, -rays[:, 2])
        xy = tf.multiply(rays[:, :2], dist[:, None])
        xy = tf.add(xy, tf.concat([pos_x, pos_y], axis=0)[None, :])

        ri = tf.divide(tf.multiply(self.ref_shape[0], tf.subtract(self.ref_origin[0], xy[:, 1])), self.ref_dim[0])
        ci = tf.divide(tf.multiply(self.ref_shape[1], tf.add(self.ref_origin[1], xy[:, 0])), self.ref_dim[1])

        ri = tf.maximum(0., tf.minimum(ri, self.ref_shape[0] - 1.))
        ci = tf.maximum(0., tf.minimum(ci, self.ref_shape[1] - 1.))

        cri = tf.stack([ci, ri], axis=1)
        values = tfa_image.resampler(self.ref[None, :, :, None], cri[None, :, :])

        # print("Ray-Tracing time:", time.time() - t1, "sec")
        tf.print("Ray-Tracing time:", tf.timestamp() - t0, "sec")
        return values[0, :, 0]

    def make_image(self, values, plot=False, title="Image"):
        shape = (self.height.numpy(), self.width.numpy())
        img = np.zeros(shape[0] * shape[1])
        img[self.r_idx.numpy()] = values.numpy()
        img = img.reshape(shape)

        if plot:
            plt.figure("Rendered " + title, (16, 9))
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

    def __call__(self, inputs):
        return self.trace_rays(inputs)

    def update(self, guess, gt):
        i = self.optimizer.iterations
        tf.print("all_vars %d:" % i, self.all_vars)

        with tf.GradientTape() as tape:
            loss = tf.keras.losses.mean_squared_error(self(guess), gt)

        t0 = tf.timestamp()
        # t1 = time.time()
        grads = tape.gradient(loss, self.all_vars)
        # print("Gradients time:", time.time() - t1, "sec")
        tf.print("Gradients %d time:" % i, tf.timestamp() - t0, "sec")

        self.optimizer.apply_gradients(zip(grads, self.all_vars))

    def optimize(self, guess, gt, epoch=50, rate=0.01, decay=1., plot=False, skip=5):
        if plot:
            gt_img = self.make_image(gt, plot=plot, title="Ground Truth")
            guess_img = self.make_image(self(guess), plot=plot, title="Initial Guess")
            self.overlay(gt_img, guess_img, "Before")

        print("\nOptimizing...")
        t0 = time.time()

        for i in range(epoch):
            self.optimizer.learning_rate = rate * math.exp(-decay * i / epoch)

            ti = time.time()
            self.update(guess, gt)
            time.sleep(0.01)
            print("Iteration %d:" % i, time.time() - ti, "sec", "\n")

            if plot and i % skip == 0:
                self.overlay(gt_img, self.make_image(self(guess)), "Progress")
                plt.pause(0.05)

        tf.print("opt_vars:", self.all_vars)
        print("Optimization time:", time.time() - t0, "sec")

        if plot:
            opt_img = self.make_image(self(guess), plot=plot, title="Optimized")
            self.overlay(gt_img, opt_img, "After")

        adjustments = [float((var * scale).numpy()) for var, scale in zip(self.all_vars, self.all_scales)]
        return [g + a for g, a in zip(guess, adjustments)]


def load_reference(filename, plot=False):
    ref = np.load(filename + ".npy")
    ref = ref[2::5, 2::5]  # make it 1px/in (faster optimization)
    print("reference:", ref.shape, ref.dtype)

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


if __name__ == "__main__":
    reference_filename = "../baseball_field/baseball_field_segmentation_5px_per_in"

    ref = load_reference(reference_filename, plot=False)
    mask = gen_mask(1920, 1080, plot=False)

    cal = Calibrator(ref, mask, scales=[1, 100, 100, 10, 10, 1], plot=False)
    # cal.build(tf.TensorShape([6]))
    # cal.compile(loss="mse", optimizer="nadam")

    calib_true = (5, 400, 400, 50, 180 + 45, 5)
    calib_guess = (4.9, 410, 395, 45, 180 + 46, 4.9)
    # calib_guess = (4.5, 450, 390, 65, 180 + 47, 4.9)

    calib_opt = cal.optimize(calib_guess, cal(calib_true), epoch=30, rate=0.02, decay=1.5, plot=True, skip=5)

    print("Optimal calibration:", calib_opt)

    plt.show()
