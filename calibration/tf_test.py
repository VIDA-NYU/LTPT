import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.representation import grid
from tensorflow_graphics.geometry.representation import ray
from tensorflow_graphics.geometry.representation import vector
from tensorflow_graphics.rendering.camera import orthographic
from tensorflow_graphics.math import spherical_harmonics
from tensorflow_graphics.math import math_helpers as tf_math

tf.compat.v1.enable_eager_execution()

#@title Controls { vertical-output: false, run: "auto" }
max_band = 2  #@param { type: "slider", min: 0, max: 10 , step: 1 }

#########################################################################
# This cell creates a lighting function which we approximate with an SH #
#########################################################################

def image_to_spherical_coordinates(image_width, image_height):
  pixel_grid_start = np.array((0, 0), dtype=type)
  pixel_grid_end = np.array((image_width - 1, image_height - 1), dtype=type)
  pixel_nb = np.array((image_width, image_height))
  pixels = grid.generate(pixel_grid_start, pixel_grid_end, pixel_nb)
  normalized_pixels = pixels / (image_width - 1, image_height - 1)
  spherical_coordinates = tf_math.square_to_spherical_coordinates(
      normalized_pixels)
  return spherical_coordinates


def light_function(theta, phi):
  theta = tf.convert_to_tensor(theta)
  phi = tf.convert_to_tensor(phi)
  zero = tf.zeros_like(theta)
  return tf.maximum(zero,
                    -4.0 * tf.sin(theta - np.pi) * tf.cos(phi - 2.5) - 3.0)


light_image_width = 30
light_image_height = 30
type = np.float64

# Builds the pixels grid and compute corresponding spherical coordinates.
spherical_coordinates = image_to_spherical_coordinates(light_image_width,
                                                       light_image_height)
theta = spherical_coordinates[:, :, 1]
phi = spherical_coordinates[:, :, 2]

# Samples the light function.
sampled_light_function = light_function(theta, phi)
ones_normal = tf.ones_like(theta)
spherical_coordinates_3d = tf.stack((ones_normal, theta, phi), axis=-1)
samples_direction_to_light = tf_math.spherical_to_cartesian_coordinates(
    spherical_coordinates_3d)

# Samples the SH.
l, m = spherical_harmonics.generate_l_m_permutations(max_band)
l = tf.convert_to_tensor(l)
m = tf.convert_to_tensor(m)
l_broadcasted = tf.broadcast_to(l, [light_image_width, light_image_height] +
                                l.shape.as_list())
m_broadcasted = tf.broadcast_to(m, [light_image_width, light_image_height] +
                                l.shape.as_list())
theta = tf.expand_dims(theta, axis=-1)
theta_broadcasted = tf.broadcast_to(
    theta, [light_image_width, light_image_height, 1])
phi = tf.expand_dims(phi, axis=-1)
phi_broadcasted = tf.broadcast_to(phi, [light_image_width, light_image_height, 1])
sh_coefficients = spherical_harmonics.evaluate_spherical_harmonics(
    l_broadcasted, m_broadcasted, theta_broadcasted, phi_broadcasted)
sampled_light_function_broadcasted = tf.expand_dims(
    sampled_light_function, axis=-1)
sampled_light_function_broadcasted = tf.broadcast_to(
    sampled_light_function_broadcasted,
    [light_image_width, light_image_height] + l.shape.as_list())

# Integrates the light function times SH over the sphere.
projection = sh_coefficients * sampled_light_function_broadcasted * 4.0 * math.pi / (
    light_image_width * light_image_height)
light_coeffs = tf.reduce_sum(projection, (0, 1))

# Reconstructs the image.
reconstructed_light_function = tf.squeeze(
    vector.dot(sh_coefficients, light_coeffs))

print(
    "average l2 reconstruction error ",
    np.linalg.norm(sampled_light_function - reconstructed_light_function) /
    (light_image_width * light_image_height))

vmin = np.minimum(
    np.amin(np.minimum(sampled_light_function, reconstructed_light_function)),
    0.0)
vmax = np.maximum(
    np.amax(np.maximum(sampled_light_function, reconstructed_light_function)),
    1.0)
# Plots results.
plt.figure(figsize=(10, 10))
ax = plt.subplot("131")
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.grid(False)
ax.set_title("Original lighting function")
_ = ax.imshow(sampled_light_function, vmin=vmin, vmax=vmax)
ax = plt.subplot("132")
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.grid(False)
ax.set_title("Spherical Harmonics approximation")
_ = ax.imshow(reconstructed_light_function, vmin=vmin, vmax=vmax)
ax = plt.subplot("133")
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.grid(False)
ax.set_title("Difference")
_ = ax.imshow(
    np.abs(reconstructed_light_function - sampled_light_function),
    vmin=vmin,
    vmax=vmax)

plt.show()