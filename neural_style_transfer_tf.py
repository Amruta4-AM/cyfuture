
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow.keras.preprocessing.image as image

# Device configuration (TensorFlow automatically handles GPU if available)
imsize = 512  # Use 128 for CPU to reduce computation

# Image loading and preprocessing
def load_image(image_path):
    img = image.load_img(image_path, target_size=(imsize, imsize))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.applications.vgg19.preprocess_input(img)  # VGG19 preprocessing
    return tf.convert_to_tensor(img)

# Load content and style images
content_image = load_image("content.jpg")
style_image = load_image("style.jpg")

# Initialize generated image (start with content image)
generated_image = tf.Variable(content_image, dtype=tf.float32)

# Load VGG19 model (excluding top layers)
vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
vgg.trainable = False

# Define content and style layers
content_layers = ["block4_conv2"]  # Layer for content
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]  # Layers for style

# Extract layer outputs
outputs = [vgg.get_layer(name).output for name in (content_layers + style_layers)]
model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

# Loss weights
content_weight = 1e4
style_weight = 1e8
tv_weight = 1e-2

# Content loss
def content_loss(content_features, generated_features):
    return tf.reduce_mean(tf.square(content_features - generated_features))

# Style loss (using Gram matrix)
def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def style_loss(style_features, generated_features):
    style_gram = gram_matrix(style_features)
    generated_gram = gram_matrix(generated_features)
    return tf.reduce_mean(tf.square(style_gram - generated_gram))

# Total variation loss for smoothness
def total_variation_loss(img):
    x_deltas = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_deltas = img[:, 1:, :, :] - img[:, :-1, :, :]
    return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))

# Compute losses
def compute_loss(model, generated_image, content_image, style_image):
    # Forward pass
    generated_outputs = model(generated_image)
    content_outputs = model(content_image)
    style_outputs = model(style_image)

    # Split outputs
    content_gen_features = generated_outputs[:len(content_layers)]
    style_gen_features = generated_outputs[len(content_layers):]
    content_features = content_outputs[:len(content_layers)]
    style_features = style_outputs[len(content_layers):]

    # Compute losses
    c_loss = 0
    s_loss = 0
    for c_f, c_g in zip(content_features, content_gen_features):
        c_loss += content_loss(c_f, c_g)
    for s_f, s_g in zip(style_features, style_gen_features):
        s_loss += style_loss(s_f, s_g)

    # Total variation loss
    tv_loss = total_variation_loss(generated_image)

    # Combine losses
    total_loss = (content_weight * c_loss +
                  style_weight * s_loss +
                  tv_weight * tv_loss)
    return total_loss, c_loss, s_loss, tv_loss

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

# Training step
@tf.function
def train_step(generated_image, content_image, style_image):
    with tf.GradientTape() as tape:
        total_loss, c_loss, s_loss, tv_loss = compute_loss(
            model, generated_image, content_image, style_image
        )
    gradients = tape.gradient(total_loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])
    # Clip pixel values to [0, 255]
    generated_image.assign(tf.clip_by_value(generated_image, 0, 255))
    return total_loss, c_loss, s_loss

# Run style transfer
iterations = 1000
for step in range(iterations):
    total_loss, c_loss, s_loss = train_step(generated_image, content_image, style_image)
    if step % 100 == 0:
        print(f"Step {step}: Total Loss: {total_loss:.4f}, "
              f"Content Loss: {c_loss:.4f}, Style Loss: {s_loss:.4f}")

# Post-process and save output
output = generated_image.numpy().squeeze()
output = np.clip(output, 0, 255).astype("uint8")
output_img = Image.fromarray(output)
output_img.save("output.jpg")
print("Image saved as output.jpg")