# My deep learning machine

... is a twin GPU machine running Linux. I run TensorFlow and PyTorch
on it. Below is a listing of the hardware I put together for it
and the software installed on it.

## Hardware

* __CPU:__ Intel Core i7-6850k 3.6GHz 6-Core processor.
    * Broadwell architecture, LGA 2011-v3 socket.
    * Most importantly for deep learning, it has 40 PCIe lanes. If running two GPUs in parallel 2x16 configuration, the CPU can be a bottleneck and the CPU needs to
  have at least 32 PCIe lanes for this. Plus, I'm using an NVME SSD
  storage to enable fast reading and writing of data to disk and that
  takes up 4 more lanes.
* __GPU:__ Two of EVGA GeForce GTX 1080 Ti 11GB FTW3 Elite video cards.
    * Pascal architecture.
    * Clock 1569 MHz - boostable to 1683 MHz
    * 11 GB of GDDR5X VRAM
    * Compute capability of 6.1
* __Motherboard:__ ASRock - X99 Taichi ATX LGA2011-3 Motherboard
* __Memory:__ G.Skill - Ripjaws V Series 32GB (2 x 16GB) DDR4-3200 Memory
* __SSD:__ MyDigitalSSD - BPX 512GB M.2-2280 Solid State Drive
* __HDD:__ Seagate - Barracuda 3TB 3.5" 7200RPM Internal Hard Drive
* __CPU Cooler:__ Thermaltake - Frio Silent 12 55.9 CFM CPU Cooler
* __Power Supply:__ EVGA - SuperNOVA G3 1000W 80+ Gold Certified Fully-Modular ATX Power Supply
* __Case:__ Corsair - 780T ATX Full Tower Case

## What it's running

* Ubuntu 18.04
* Cuda 9.0.176
* NVIDIA Driver v390.87
* CuDNN 7.0
* TensorFlow 1.10.0
* Keras 2.2.2
* PyTorch version 0.4.1.post2
* Python 3.6 and 2.7
* Anaconda 3 for package management
