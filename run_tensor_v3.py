import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
import os
import pickle

# 2. Function to load CIFAR-10 data
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']

def load_cifar10(data_path):
    x_train = []
    y_train = []
    
    # Load training batches
    for i in range(1, 6):
        batch_file = os.path.join(data_path, f'data_batch_{i}')
        data, labels = load_cifar10_batch(batch_file)
        x_train.append(data)
        y_train.extend(labels)
    
    x_train = np.concatenate(x_train)
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Load test batch
    test_file = os.path.join(data_path, 'test_batch')
    x_test, y_test = load_cifar10_batch(test_file)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return (x_train, np.array(y_train)), (x_test, np.array(y_test))


def build_tensorrt_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 28  # 256MB
    
    engine = builder.build_engine(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

# 5. Run inference with TensorRT
def run_tensorrt_inference(engine_path, input_data):
    logger = trt.Logger(trt.Logger.WARNING)
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    
    # Allocate memory for input and output
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    # Copy input data to device
    np.copyto(h_input, input_data.ravel())
    cuda.memcpy_htod(d_input, h_input)
    
    # Run inference
    context.execute_v2(bindings=[int(d_input), int(d_output)])
    
    # Copy results back to host
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output

# Example usage
if __name__ == "__main__":
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = load_cifar10("cifar-10-batches-py")
    
    # Convert model
    #convert_tflite_to_onnx("model.tflite", "model.onnx")
    #build_tensorrt_engine("model.onnx", "model.trt")
    
    # Run inference
    sample_input = x_test[0:1]  # Take first test image
    # Warm-up run
    _ = run_tensorrt_inference("cifar10_model.engine", sample_input)
    
    # Measure inference time
    start_time = time.time()
    result = run_tensorrt_inference("cifar10_model.engine", sample_input)
    end_time = time.time()
    
    print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")
    print("Prediction:", np.argmax(result))
    #result = run_tensorrt_inference("model.trt", sample_input)
    #print("Prediction:", np.argmax(result))
