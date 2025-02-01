import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load TRT engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Allocate memory for input and output
        self.host_inputs = []
        self.host_outputs = []
        self.cuda_inputs = []
        self.cuda_outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(cuda_mem))
            
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def run_inference(self, input_data):
        # Copy input data to host buffer
        np.copyto(self.host_inputs[0], input_data.ravel())
        
        # Transfer input data to GPU
        cuda.memcpy_htod(self.cuda_inputs[0], self.host_inputs[0])
        
        # Run inference
        self.context.execute_v2(bindings=self.bindings)
        
        # Transfer predictions back from GPU
        cuda.memcpy_dtoh(self.host_outputs[0], self.cuda_outputs[0])
        
        return self.host_outputs[0]

# Example usage
def main():
    # Initialize TensorRT inference
    trt_inference = TensorRTInference("model.trt")
    
    # Load your test data (example with CIFAR-10)
    test_image = np.random.random((1, 32, 32, 3)).astype(np.float32)  # Replace with actual test data
    
    # Warm-up run
    _ = trt_inference.run_inference(test_image)
    
    # Measure inference time
    start_time = time.time()
    result = trt_inference.run_inference(test_image)
    end_time = time.time()
    
    print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")
    print("Prediction:", np.argmax(result))

if __name__ == "__main__":
    main()
