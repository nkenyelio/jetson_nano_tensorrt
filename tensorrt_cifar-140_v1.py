import tensorflow as tf
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt


# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Convert to TFLite
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()

# Save the TFLite model
#with open("mnist_model.tflite", "wb") as f:
#    f.write(tflite_model)


# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

# Select the first 50 images for inference
x_test = x_test[:50]
y_test = y_test[:50]

# Load the TensorRT engine
with open("cifar10_model.engine", "rb") as f:
    engine_data = f.read()
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)

# Create an execution context
context = engine.create_execution_context()

# Get input and output binding details
input_shape = engine.get_binding_shape(0)
output_shape = engine.get_binding_shape(1)

# Allocate memory for inputs and outputs
input_memory = cuda.mem_alloc(np.prod(input_shape) * 4)  # 4 bytes per float32
output_memory = cuda.mem_alloc(np.prod(output_shape) * 4)

def preprocess_image(image):
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

results = []
for image in x_test:
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Copy input data to GPU
    cuda.memcpy_htod(input_memory, preprocessed_image)

    # Run inference
    context.execute_v2(bindings=[int(input_memory), int(output_memory)])

    # Copy output data from GPU
    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output_data, output_memory)

    results.append(output_data)
predicted_labels = [np.argmax(result) for result in results]
correct_predictions = np.sum(np.array(predicted_labels) == y_test)
accuracy = correct_predictions / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {predicted_labels[i]}, Actual: {y_test[i]}")
    plt.show()
