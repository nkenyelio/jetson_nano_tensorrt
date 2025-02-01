import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
import pickle
import os
from tqdm import tqdm
import gc

class CIFAR10Loader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 
                           'data_batch_4', 'data_batch_5', 'test_batch']
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

    def load_batch(self, filename):
        path = os.path.join(self.data_dir, filename)
        with open(path, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        return (dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0,
                np.array(dict[b'labels']))

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        print("Loading TensorRT engine...")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        self.host_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        self.host_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        self.cuda_input = cuda.mem_alloc(self.host_input.nbytes)
        self.cuda_output = cuda.mem_alloc(self.host_output.nbytes)
        
        self.bindings = [int(self.cuda_input), int(self.cuda_output)]

    def run_inference(self, input_data):
        np.copyto(self.host_input, input_data.ravel())
        cuda.memcpy_htod(self.cuda_input, self.host_input)
        self.context.execute_v2(bindings=self.bindings)
        cuda.memcpy_dtoh(self.host_output, self.cuda_output)
        return self.host_output.copy()

    def __del__(self):
        del self.context
        del self.engine
        del self.runtime

def run_full_inference():
    try:
        # Initialize data loader and inference engine
        data_loader = CIFAR10Loader("cifar-10-batches-py")
        trt_inference = TensorRTInference("cifar10_model.engine")
        
        # Statistics containers
        total_images = 0
        correct_predictions = 0
        inference_times = []
        confusion_matrix = np.zeros((10, 10), dtype=int)
        class_correct = np.zeros(10)
        class_total = np.zeros(10)
        
        print("\nStarting full CIFAR-10 inference...")
        
        # Process each batch
        for batch_file in data_loader.batch_files:
            print(f"\nProcessing {batch_file}...")
            images, labels = data_loader.load_batch(batch_file)
            
            # Process each image in the batch with progress bar
            for idx in tqdm(range(len(images)), desc="Images"):
                try:
                    image = images[idx]
                    true_label = labels[idx]
                    
                    # Run inference with timing
                    start_time = time.time()
                    output = trt_inference.run_inference(image)
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Process results
                    predicted_class = np.argmax(output)
                    inference_times.append(inference_time)
                    
                    # Update statistics
                    if predicted_class == true_label:
                        correct_predictions += 1
                        class_correct[true_label] += 1
                    class_total[true_label] += 1
                    confusion_matrix[true_label][predicted_class] += 1
                    total_images += 1
                    
                except Exception as e:
                    print(f"\nError processing image {idx} in {batch_file}: {str(e)}")
            
            # Force garbage collection between batches
            gc.collect()
        
        # Calculate and print comprehensive statistics
        print("\n" + "="*50)
        print("CIFAR-10 Inference Results")
        print("="*50)
        
        # Overall statistics
        print("\nOverall Statistics:")
        print(f"Total images processed: {total_images}")
        print(f"Overall accuracy: {(correct_predictions/total_images)*100:.2f}%")
        print(f"Average inference time: {np.mean(inference_times):.2f}ms")
        print(f"Min inference time: {np.min(inference_times):.2f}ms")
        print(f"Max inference time: {np.max(inference_times):.2f}ms")
        
        # Per-class statistics
        print("\nPer-class Accuracy:")
        for i in range(10):
            accuracy = (class_correct[i]/class_total[i])*100
            print(f"{data_loader.label_names[i]:<10}: {accuracy:.2f}%")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("True\\Pred", end=" ")
        for i in range(10):
            print(f"{i:4}", end=" ")
        print("\n" + "-"*55)
        
        for i in range(10):
            print(f"{i:9}", end=" ")
            for j in range(10):
                print(f"{confusion_matrix[i][j]:4}", end=" ")
            print()
        
        return {
            'total_images': total_images,
            'accuracy': correct_predictions/total_images,
            'avg_inference_time': np.mean(inference_times),
            'confusion_matrix': confusion_matrix,
            'class_accuracy': class_correct/class_total
        }
        
    except Exception as e:
        print(f"\nError in full inference process: {str(e)}")
        return None

if __name__ == "__main__":
    results = run_full_inference()
    
    if results:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed. Check error messages above.")
