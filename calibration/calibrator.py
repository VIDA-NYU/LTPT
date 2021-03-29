import os
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pymongo import MongoClient
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

        self.build_scaled_vars(defaults or [0] * 7, scales or [100, 100, 10, 100, 100, 100, 10])
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

        self.cam_theta = tf.Variable([0.], dtype=tf.float32)
        self.s_cam_theta = self.s_theta / 10.

        self.loc_vars = [self.pos_x, self.pos_y, self.pos_z, self.phi, self.theta]
        self.pos_vars = [self.pos_x, self.pos_y, self.pos_z]
        self.rot_vars = [self.phi, self.theta, self.cam_theta]
        self.cam_vars = [self.pos_x, self.pos_y, self.pos_z, self.cam_theta, self.dist, self.fov]

    def reset(self):
        self.losses, self.configs = [], []
        self.cam_theta.assign([0.])

        for var, default, scale in zip(self.all_vars, self.defaults, self.scales):
            var.assign([default / scale])
        tf.print("Default vars:", self.all_vars)

    def get_dist_adjustment(self, dist, fov):
        ref_tan_z = tf.tan(fov * self.to_rad / 2.)
        tan_z = tf.tan((fov + self.fov * self.s_fov) * self.to_rad / 2.)

        dist_adj = dist * (ref_tan_z - tan_z) / tan_z
        # tf.print("dist_adj:", dist_adj)

        return dist_adj

    def get_fov_adjustment(self, fov, theta):
        r = tf.sin(theta + self.cam_theta * self.s_cam_theta * self.to_rad) / tf.sin(theta)

        fov_adj = 2 * (tf.atan(r * tf.tan(fov / 2)) - (fov / 2))
        # tf.print("fov_adj:", fov_adj / self.to_rad)

        return fov_adj

    def trace_rays(self, inputs):
        pos_x, pos_y, pos_z, phi, theta, dist, fov = inputs

        pos_x = pos_x + self.pos_x * self.s_pos_x
        pos_y = pos_y + self.pos_y * self.s_pos_y
        pos_z = pos_z + self.pos_z * self.s_pos_z

        dist_adj = self.get_dist_adjustment(dist, fov)
        fov = (fov + self.fov * self.s_fov) * self.to_rad
        dist = dist + self.dist * self.s_dist + dist_adj

        phi = (phi + self.phi * self.s_phi) * self.to_rad
        theta = (theta + self.theta * self.s_theta) * self.to_rad

        fov_adj = self.get_fov_adjustment(fov, theta)
        fov = fov + fov_adj
        theta = theta + self.cam_theta * self.s_cam_theta * self.to_rad
        tf.print("Theta:", theta / self.to_rad, "Dist:", dist, "Fov:", fov / self.to_rad)

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
        config = [b + a for b, a in zip(base, adjustments)]
        base_dist, base_fov = base[-2], base[-1]
        config_fov, config_theta = config[-1], config[-3]
        config.append(float((self.cam_theta * self.s_cam_theta).numpy()))
        config.append(float(self.get_dist_adjustment(base_dist, base_fov).numpy()))
        config.append(float((self.get_fov_adjustment(config_fov * self.to_rad, config_theta * self.to_rad) / self.to_rad).numpy()))
        return tuple(config)

    def set_config(self, config, base):
        adjustments = [c - b for c, b in zip(config[:-3] if len(config) == 10 else config, base)]
        self.cam_theta.assign([config[-3] / self.s_cam_theta])

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
        self.best_loss = self.losses[best]
        print("Best config %d:" % best, self.best_config)
        print("Min loss so far:", self.losses[best])
        self.set_config(self.best_config, guess)

        ret = list(self.best_config[:-3])
        ret[-3] += self.best_config[-3]  # cam_theta
        ret[-2] += self.best_config[-2]  # dist_adj
        ret[-1] += self.best_config[-1]  # fov_adj
        ret = tuple(ret)

        if last:
            self.reset()  # To plot with opt_config instead of guess + vars
            self.best_img = self.make_image(self(ret), plot=plot, title="Last Iteration")

            if plot:
                self.overlay(gt_img, self.best_img, "After")
                self.plot_loss()
        else:
            self.best_img = None

        return ret

    def calibrate(self, guess, gt, **kw):
        t0 = time.time()
        # cal = self.optimize(guess, gt, vars=self.cam_vars, last=True, epoch=100, rate=0.03, decay=1, **kw)
        # cal = self.optimize(guess, gt, vars=self.all_vars, last=True, epoch=10, rate=0.02, decay=2, **kw)
        cal = self.optimize(guess, gt, vars=self.all_vars, last=False, epoch=50, rate=0.01, decay=2, **kw)
        cal = self.optimize(guess, gt, vars=self.loc_vars, last=False, epoch=20, rate=0.01, decay=1, **kw)
        cal = self.optimize(guess, gt, vars=self.pos_vars, last=False, epoch=10, rate=0.01, decay=1, **kw)
        cal = self.optimize(guess, gt, vars=self.rot_vars, last=False, epoch=20, rate=0.01, decay=1, **kw)
        cal = self.optimize(guess, gt, vars=self.cam_vars, last=False, epoch=50, rate=0.01, decay=1, **kw)
        cal = self.optimize(guess, gt, vars=self.all_vars, last=True, epoch=10, rate=0.003, decay=1, **kw)
        print("Total optimization time:", time.time() - t0, "sec")
        return cal, float(self.best_loss), self.best_img


def to_gray(img):
    img[np.nonzero(img)] = 1
    img *= np.array([1, 2, 4], dtype=np.uint8)[None, None, :]

    gray = np.zeros(img.shape[:2], dtype=np.uint8)
    gray[...] = np.sum(img, axis=2, dtype=np.uint8)
    correspondence = np.array([0, 1, 2, 6, 3, 5, 7, 4], dtype=np.uint8)
    gray[...] = correspondence[gray.ravel()].reshape(img.shape[:2])

    return gray


def from_gray(gray):
    img = np.zeros((*gray.shape, 3), dtype=np.uint8)
    colors = np.array([[0, 0, 0],       # background
                       [255, 0, 0],     # dirt
                       [0, 255, 0],     # grass
                       [0, 0, 255],     # home plate
                       [255, 255, 255], # lines
                       [255, 0, 255],   # pitcher mound
                       [255, 255, 0],   # people
                       [0, 255, 255]],  # bases
                       dtype=np.uint8)

    img.reshape((-1, 3))[...] = colors[gray.ravel(), :]

    return img


def to_opt(img):
    # mapping = np.array([0, 2, 1, 3, 3, 4, 0, 3], dtype=np.uint8) # dirt, grass, lines and separate pitcher mound
    mapping = np.array([0, 2, 1, 3, 3, 2, 0, 3], dtype=np.uint8)  # dirt, grass and lines
    # mapping = np.array([0, 2, 0, 3, 3, 2, 0, 3], dtype=np.uint8)  # dirt & lines only
    return mapping[img.ravel()].reshape(img.shape[:2])


def peter_to_gray(img, annot=False):
    # plt.figure("Img")
    # plt.imshow(img)

    gray = np.zeros(img.shape[:2], dtype=np.uint32)
    gray[...] = np.sum(img.astype(np.uint32), axis=2, dtype=np.uint32)

    correspondence = np.zeros(255*3+1, dtype=np.uint8)

    if annot:
        correspondence[255 + 255] = 1
        correspondence[81 + 81] = 1
        correspondence[107 + 142 + 35] = 2
        correspondence[255 + 255 + 255] = 4
    else:
        # old
        # correspondence[128 + 64 + 128] = 1
        # correspondence[70 + 70 + 70] = 2
        # new
        correspondence[128 + 64 + 128] = 1
        correspondence[152 + 251 + 152] = 2
        correspondence[220 + 20 + 60] = 6

    gray[...] = correspondence[gray.ravel()].reshape(img.shape[:2])

    # plt.figure("Gray")
    # plt.imshow(gray)
    # plt.show()

    return gray


def cache_reference(filename, downsample=True, plot=False):
    ref = cv2.imread(filename + ".png")[:, :, ::-1]

    if downsample:
        ref = ref[2::5, 2::5]  # make it 1px/in (faster optimization)

    print(ref.shape, ref.dtype)

    if plot:
        plt.figure("Color Reference")
        plt.imshow(ref)

    gray = to_gray(ref)
    np.save(filename + ".npy", gray)

    if plot:
        plt.figure("Gray Reference")
        plt.imshow(gray)

        plt.figure("Colored Reference")
        plt.imshow(from_gray(gray))


def load_reference(filename, optimize=True, downsample=False, plot=False):
    ref = np.load(filename + ".npy")

    if downsample:
        ref = ref[2::5, 2::5]  # make it 1px/in (faster optimization)

    # print("reference:", ref.shape, ref.dtype)

    if optimize:
        ref = to_opt(ref)

    if plot:
        plt.figure("Optimization Reference", (16, 9))
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


def validate(plot=True):
    reference_filename = "../baseball_field/baseball_field_segmentation_5px_per_in_2"

    ref = load_reference(reference_filename, plot=False)
    mask = gen_mask(1920, 1080, plot=False)

    cal = Calibrator(ref, mask, plot=plot)

    # Behind Pitcher
    # calib_true = (45, 45, 5, 180 + 45, 5, 400, 5)
    # calib_guess = (55, 50, 7, 180 + 44, 6, 350, 4.5)

    # Behind Pitcher (Zoom)
    # calib_true = (45, 45, 5, 180 + 45, 7, 400, 5)
    # calib_guess = (45, 45, 5, 180 + 45, 7, 350, 4.5)

    # Pitcher Side (Right Handed)
    # calib_true = (45, 45, 10, 90 + 55, 7, 200, 10)
    # calib_guess = (75, 30, 6, 90 + 50, 6, 250, 8)

    # Batter Side (Right Handed)
    # calib_true = (0, 0, 5, -55, 4, 100, 15)
    # calib_guess = (5, 10, 3, -40, 3, 122, 12)

    # Behind Batter
    calib_true = (40, 40, 0, 45, 20, 300, 25)
    calib_guess = (50, 60, 10, 50, 15, 330, 22)

    calib_opt, _, _ = cal.calibrate(calib_guess, cal(calib_true), plot=plot)
    print("\nOptimal calibration:", calib_opt)


def test(plot=True):
    reference_filename = "../baseball_field/baseball_field_segmentation_5px_per_in_2"
    # segmentation_filename = "../baseball_field/baseball_field_example_1_segmented.png"
    segmentation_filename = "../baseball_field/baseball_field_example_2_segmented.png"
    # segmentation_filename = "../baseball_field/baseball_field_example_weird_segmented.png"

    ref = load_reference(reference_filename, plot=False)
    seg = cv2.imread(segmentation_filename)[:, :, ::-1]

    gt = to_opt(to_gray(seg))
    mask = gt > 0
    gt = gt.ravel()[mask.ravel()]

    cal = Calibrator(ref, mask, plot=plot)

    calib_guess = (50, 40, 10, 50, 15, 330, 22)
    tf_gt = tf.convert_to_tensor(value=gt, dtype=tf.float32)

    calib_opt, _, _ = cal.calibrate(calib_guess, tf_gt, plot=plot)
    print("\nOptimal calibration:", calib_opt)


def generate_single(path, id, view, side, out_path=None, plot=True):
    reference_filename = "../baseball_field/baseball_field_segmentation_5px_per_in_2"
    ref = load_reference(reference_filename, plot=False)

    out_path = out_path or path
    filename = path + id + "_" + view + "_prediction.png"

    seg = cv2.imread(filename)[:, :, ::-1]
    gt = to_opt(to_gray(seg))
    # gt = to_opt(peter_to_gray(seg))

    mask = gt >= 0  # Ignore. Segmentations are too bad to use masks
    # mask[:mask.shape[0]//3, :] = 0  # Just trim the top thrid of the image for faster rendering
    gt = gt.ravel()[mask.ravel()]
    tf_gt = tf.convert_to_tensor(value=gt, dtype=tf.float32)

    cal = Calibrator(ref, mask, plot=plot)

    if side == "third base":
        calib_guess = (41, 41, 5, 90 + 35, 1, 100, 6)
    else:
        calib_guess = (41, 41, 5, -35, 1, 100, 6)

    calib_opt, loss, img = cal.calibrate(calib_guess, tf_gt, plot=plot)
    print("\nOptimal calibration:", calib_opt)

    if img is not None:
        img *= 100
        img = img.astype(np.uint8)
        cv2.imwrite(out_path + id + "_" + view + "_calibration.png", np.stack([img, img, img], axis=2))

    with open(out_path + id + "_" + view + ".json", "w") as f:
        json.dump({"calib": calib_opt,
                   "loss": loss,
                   "calib_hint": "pox_x (feet), pox_y (feet), pox_z (feet), phi (deg), theta (deg), dist (feet), fov (deg)",
                   "loss_hint": "MSE over pixel differences"}, f, indent=4)


def generate_all(path, out_path=None, plot=False):
    db = MongoClient("mongodb+srv://yurii:yuriimongo@ltpt.qsvio.mongodb.net").ltpt
    videos = list(db["videos"].find({}))
    print(len(videos), "videos")
    print(videos[0])

    pitcher_vids = {str(vid["_id"]): {"view": vid["view"], "pitcher_height": vid["pitcher_height"],
                                      "predicted_side": vid["predicted_side"]} for vid in videos if "pitcher_height" in vid}

    print(len(pitcher_vids), "pitcher videos")
    print(next(iter(pitcher_vids.items())))

    out_path = out_path or path

    # id = "603ffd0a71a81c8b9eae028b"
    # id = "603ffd0c71a81c8b9eae02ab"
    # for i, (id, vid) in enumerate([(id, pitcher_vids[id])]):
    for i, (id, vid) in enumerate(pitcher_vids.items()):
        view, side = vid["view"], vid["predicted_side"]

        filename = path + id + "_" + view + "_prediction.png"

        if not os.path.exists(filename):
            print("Missing:", filename)
            continue

        if view != "C":
            print("Wrong view:", filename)
            continue

        # Slice HERE
        if i > 2:
            break

        generate_single(path, id, view, side, out_path=out_path, plot=plot)


if __name__ == "__main__":
    # cache_reference("../baseball_field/baseball_field_segmentation_5px_per_in_2", plot=True)
    # ref = load_reference("../baseball_field/baseball_field_segmentation_5px_per_in_2", plot=True)
    # plt.show()
    # exit()

    # validate(plot=True)
    # test(plot=False)

    # path = "../some_segmented/"
    path = "../best_images_2203/"

    out_path = "../pitcher_images/"
    # out_path = path

    generate_all(path, out_path=out_path, plot=True)

    plt.show()
