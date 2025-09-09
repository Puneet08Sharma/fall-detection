import tensorflow as tf

h5_model_path = "models/fall_detection_model.h5"
tflite_model_path = "models/fall_detection_model.tflite"

model = tf.keras.models.load_model(h5_model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_new_converter = True
tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"✅ TFLite model saved at {tflite_model_path}")
