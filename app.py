import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}

model = keras.models.load_model("brain_tumor_effnet.keras")

# Split the model into: base (EfficientNet) -> pooling -> dense head
base = [l for l in model.layers if isinstance(l, keras.Model)][0]
gap = [l for l in model.layers if isinstance(l, layers.GlobalAveragePooling2D)][0]
dense = model.layers[-1]


def gradcam(arr):
    img = tf.convert_to_tensor(arr, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out = base(img, training=False)
        tape.watch(conv_out)
        preds = dense(gap(conv_out))
        score = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), preds.numpy()[0]


def predict(image):
    if image is None:
        return None, None
    img = Image.fromarray(image).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)

    cam, probs = gradcam(arr)

    # heatmap overlay
    cam_big = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    ) / 255.0
    heat = (cm.jet(cam_big)[..., :3] * 255).astype(np.uint8)
    overlay = (0.6 * np.array(img) + 0.4 * heat).astype(np.uint8)

    scores = {LABELS[c]: float(p) for c, p in zip(CLASS_NAMES, probs)}
    return scores, overlay


DISCLAIMER = (
    "**Educational project only. Not a medical device and not for diagnosis.** "
    "The model scores are not calibrated: it can be wrong while showing a high score. "
    "Glioma and meningioma are often confused. On the held-out test set it scored "
    "about 92% accuracy, and about 21% of gliomas were missed. "
    "Always rely on a qualified clinician."
)

with gr.Blocks(title="Brain MRI Classifier") as demo:
    gr.Markdown("# Brain MRI Tumor Classifier")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        inp = gr.Image(label="Upload an MRI slice", type="numpy")
        with gr.Column():
            out_label = gr.Label(label="Model score (not a certainty)", num_top_classes=4)
            out_cam = gr.Image(label="Grad-CAM: regions that influenced the prediction")
    btn = gr.Button("Analyze")
    btn.click(predict, inputs=inp, outputs=[out_label, out_cam])

if __name__ == "__main__":
    demo.launch()
    