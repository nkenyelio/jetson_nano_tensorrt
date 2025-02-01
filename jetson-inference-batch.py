import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
import pickle
import os

def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), dict[b'labels']

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Allocate memory for single inference
        self.host_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        self.host_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        self.cuda_input = cuda.mem_alloc(self.host_input.nbytes)
        self.cuda_output = cuda.mem_alloc(self.host_output.nbytes)
        
        self.bindings = [int(self.cuda_input), int(self.cuda_output)]

    def run_inference(self, input_data):
        # Preprocess input data (normalize if needed)
        input_data = input_data.astype(np.float32) / 255.0  # Normalize to [0,1]
        
        # Copy input data to host buffer
        np.copyto(self.host_input, input_data.ravel())
        
        # Transfer input data to GPU
        cuda.memcpy_htod(self.cuda_input, self.host_input)
        
        # Run inference
        self.context.execute_v2(bindings=self.bindings)
        
        # Transfer predictions back from GPU
        cuda.memcpy_dtoh(self.host_output, self.cuda_output)
        
        return self.host_output.copy()

def process_batch_images():
    try:
        # Load first test batch from CIFAR-10
        test_data, test_labels = load_cifar10_batch("cifar-10-batches-py/data_batch_1")
        
        # Take first 20 images
        images = test_data[:20]
        labels = test_labels[:20]
        
        # Initialize TensorRT inference
        trt_inference = TensorRTInference("cifar10_model.engine")
        
        # Store results
        results = []
        inference_times = []
        
        print("\nProcessing 20 images:")
        print("-" * 50)
        
        # Process each image
        for i in range(20):
            try:
                # Prepare single image
                image = images[i]
                true_label = labels[i]
                
                # Run inference and measure time
                start_time = time.time()
                output = trt_inference.run_inference(image)
                inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Get prediction
                predicted_class = np.argmax(output)
                
                # Store results
                results.append({
                    'image_id': i,
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'confidence': float(output[predicted_class]),
                    'inference_time': inference_time
                })
                
                # Print progress
                print(f"Image {i+1:2d}/20: "
                      f"Predicted={predicted_class:1d}, "
                      f"Actual={true_label:1d}, "
                      f"Time={inference_time:.2f}ms")
                
                inference_times.append(inference_time)
                
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 50)
        print(f"Average inference time: {np.mean(inference_times):.2f}ms")
        print(f"Min inference time: {np.min(inference_times):.2f}ms")
        print(f"Max inference time: {np.max(inference_times):.2f}ms")
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in results if r['true_label'] == r['predicted_class'])
        accuracy = correct_predictions / len(results)
        print(f"Accuracy on 20 images: {accuracy*100:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return None

if __name__ == "__main__":
    results = process_batch_images()
    
    if results:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed. Check error messages above.")
