import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import sys
import gc

class TensorRTInference:
    def __init__(self, engine_path, verbose=True):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.verbose = verbose
        
        try:
            # Force garbage collection before allocating new memory
            gc.collect()
            cuda.init()
            
            # Print available memory (if verbose)
            if self.verbose:
                free, total = cuda.mem_get_info()
                print(f"Free GPU memory: {free/1024**2:.2f} MB")
                print(f"Total GPU memory: {total/1024**2:.2f} MB")
            
            # Load engine file
            if self.verbose:
                print("Loading engine file...")
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            if self.verbose:
                print("Creating runtime...")
            self.runtime = trt.Runtime(self.logger)
            
            if self.verbose:
                print("Deserializing engine...")
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            
            if not self.engine:
                raise RuntimeError("Failed to create engine")
            
            if self.verbose:
                print("Creating execution context...")
            self.context = self.engine.create_execution_context()
            
            # Get input/output information
            self.input_shape = self.engine.get_binding_shape(0)
            self.output_shape = self.engine.get_binding_shape(1)
            
            if self.verbose:
                print(f"Input shape: {self.input_shape}")
                print(f"Output shape: {self.output_shape}")
            
            # Calculate memory sizes
            self.input_size = trt.volume(self.input_shape)
            self.output_size = trt.volume(self.output_shape)
            
            # Allocate memory with error checking
            try:
                if self.verbose:
                    print("Allocating host memory...")
                self.host_input = cuda.pagelocked_empty(self.input_size, dtype=np.float32)
                self.host_output = cuda.pagelocked_empty(self.output_size, dtype=np.float32)
                
                if self.verbose:
                    print("Allocating device memory...")
                self.cuda_input = cuda.mem_alloc(self.host_input.nbytes)
                self.cuda_output = cuda.mem_alloc(self.host_output.nbytes)
                
            except cuda.MemoryError as e:
                print(f"Failed to allocate memory: {str(e)}")
                raise
            
            self.bindings = [int(self.cuda_input), int(self.cuda_output)]
            
            if self.verbose:
                print("Initialization complete!")
                
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def run_inference(self, input_data):
        try:
            if input_data.size != self.input_size:
                raise ValueError(f"Input data size ({input_data.size}) doesn't match engine input size ({self.input_size})")
            
            # Copy input data to host buffer
            np.copyto(self.host_input, input_data.ravel())
            
            # Transfer input data to GPU
            cuda.memcpy_htod(self.cuda_input, self.host_input)
            
            # Run inference
            self.context.execute_v2(bindings=self.bindings)
            
            # Transfer predictions back from GPU
            cuda.memcpy_dtoh(self.host_output, self.cuda_output)
            
            return self.host_output.reshape(self.output_shape)
            
        except cuda.Error as e:
            print(f"CUDA error during inference: {str(e)}")
            raise
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            raise
        
    def __del__(self):
        # Explicit cleanup
        try:
            del self.context
            del self.engine
            del self.runtime
            gc.collect()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

def test_inference():
    try:
        # Initialize with verbose logging
        trt_inference = TensorRTInference("model.trt", verbose=True)
        
        # Create small test input
        input_shape = trt_inference.input_shape
        test_input = np.random.random(input_shape).astype(np.float32)
        
        # Run inference with error checking
        try:
            result = trt_inference.run_inference(test_input)
            print("Inference successful!")
            print(f"Output shape: {result.shape}")
            return True
        except Exception as e:
            print(f"Inference failed: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_inference()
    if not success:
        print("Test failed - check the error messages above")
        sys.exit(1)
