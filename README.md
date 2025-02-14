This repository provides an implementation of Dynamic Model Switching for edge device inference, enabling adaptive selection of quantized deep learning models based on real-time resource constraints. The system is designed to optimize inference performance on resource-limited hardware, such as NVIDIA Jetson Nano and Google Coral Dev Board.
Key Features
•	Dynamic Model Selection: switches between multiple quantized deep learning models based on available compute resources (CPU, GPU).
•	Real-time Resource Monitoring: Tracks CPU usage, memory consumption, latency, and power efficiency.
•	Benchmark-driven Switching: Uses resource-aware multi-criteria decision-making to balance accuracy and efficiency.
•	Support for Edge AI Accelerators: Runs optimized inference on TensorRT (GPU) for Jetson Nano.
•	Low Latency Inference: Ensures smooth model adaptation with minimal overhead.
How It Works
1.	Pretrained and Quantized Models: A set of optimized deep learning models are stored for different performance trade-offs.
2.	Resource Monitoring: The system continuously measures CPU, GPUutilization, inference latency, and accuracy prediction.
3.	Model Switching Logic: Based on real-time constraints, the framework dynamically selects the best model to balance accuracy and efficiency.
4.	Inference Execution: The selected model runs inference using TensorRT (GPU) on Jetson Nano.
5.	Performance Logging: Records inference performance, switching decisions, and resource usage for benchmarking and optimization.
Hardware Requirements
•	NVIDIA Jetson Nano (JetPack + TensorRT)
