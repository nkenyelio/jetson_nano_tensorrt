import numpy as np
import pickle
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import matplotlib.pyplot as plt

def load_cifar10_data(data_dir):
    # Load training data
    x_train = []
    y_train = []
    for i in range(1, 6):
        with open(os.path.join(data_dir, f"data_batch_{i}"), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            x_train.append(batch[b'data'])
            y_train.append(batch[b'labels'])
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    # Load test data
    with open(os.path.join(data_dir, "test_batch"), 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x_test = batch[b'data']
        y_test = batch[b'labels']

    # Reshape and normalize the data
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0

    return (x_train, y_train), (x_test, y_test)

# Path to the CIFAR-10 data directory
data_dir = "cifar-10-batches-py"

# Load the data
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)



# Load the TensorRT engine
with open("cifar10_model.engine", "rb") as f:
    engine_data = f.read()
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)

# Create an execution context
context = engine.create_execution_context()

# Allocate memory for inputs and outputs
input_shape = engine.get_binding_shape(0)
output_shape = engine.get_binding_shape(1)
input_memory = cuda.mem_alloc(np.prod(input_shape) * 4)  # 4 bytes per float32
output_memory = cuda.mem_alloc(np.prod(output_shape) * 4)

def preprocess_image(image):
    image = cv2.resize(image, (input_shape[2], input_shape[3]))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

results = []
for image in x_test[:50]:  # First 50 images
    preprocessed_image = preprocess_image(image)

    # Copy input data to GPU
    cuda.memcpy_htod(input_memory, preprocessed_image)

    # Run inference
    context.execute_v2(bindings=[int(input_memory), int(output_memory)])

    # Copy output data from GPU
    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output_data, output_memory)

    results.append(output_data)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i, (image, result) in enumerate(zip(x_test[:50], results)):
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    predicted_class = np.argmax(result)
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()
correct_predictions = 0
for i, (result, label) in enumerate(zip(results, y_test[:50])):
    predicted_class = np.argmax(result)
    if predicted_class == label:
        correct_predictions += 1

accuracy = correct_predictions / len(results)
print(f"Accuracy: {accuracy * 100:.2f}%")
