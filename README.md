# HI-GAN: A Hierarchical Generative Adversarial Network for Blind Denoising of Real Photographs

## Requirements
- python       3.6
- pytorch      1.0
- CUDA         8.0 or 9.0

## Dataset
1. Download train dataset from [Smartphone Image Denoising Dataset Medium Dataset[1]](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)
2. Download test dataset from
[The Darmstadt Noise Dataset (DND)[2]](https://noise.visinf.tu-darmstadt.de/), [SIDD Benchmark[1]](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php), [NAM[3]](http://snam.ml/research/ccnoise) ,[NC12[4]](http://demo.ipol.im/demo/125/)

[1] Abdelrahman Abdelhamed, Lin S., Brown M. S. A High-Quality Denoising Dataset for Smartphone Cameras. CVPR, 2018.

[2] T. Plotz, and S. Roth. Benchmarking denoising algorithms with real photographs. In 2017 IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), pages
2750–2759, 2017

[3] Seonghyeon Nam*, Youngbae Hwang*, Yasuyuki Matsushita, Seon Joo Kim. A Holistic Approach to Cross-Channel Image Noise Modeling and its Application to Image Denoising. CVPR, 2016.

[4]  M. Lebrun, M. Colom, and J.-M. Morel. The noise clinic:
A blind image denoising algorithm. In Image Processing On Line, vol. 5, pp. 1–54, 2015.

## Usage
### 1. Test
`python test.py`

Optional
- `--inp` : input folder
- `--out` : output folder
- `--JPEG` : for JPEG images such as "NAM_20_rand_pathes" testset (don't use this argument with non-JPEG images such as "DND_20_rand_patches" testset)
- `--nGpu` : number of GPU

### 2. Demo cell
`python demo_cell.py`

Optional
- `--inp-dir` : input image (select image on 'testsets/demo_cell' folder), default='testsets/demo_cell/FMD_test_mix/avg2/Confocal_BPAE_B_1.png'
- `--out` : output folder
- `--net` : choices=['N2N', 'DnCNN','UNet_ND', 'UNet_D', 'HI_GAN'], default='HI_GAN'
- `--gray` : Gray image, default=True (Create RGB image from 3 gray images)
- `--nGpu` : number of GPU

### 3. Test cell (Reproduce test example in the paper)
`python test_cell.py`

#### 4. Sample results  
DND 
<img src="images/DND.png" width="100%" height="100%">  
<img src="images/DND2.png" width="100%" height="100%">  
NAM 
<img src="images/NAM.png" width="100%" height="100%"> 
NC12 
<img src="images/NC12_1.png" width="100%" height="100%"> 
SIDD
<img src="images/SIDD.png" width="100%" height="100%"> 
FMD
<img src="images/HI_GAN_Confocal_BPAE_RGB_1.png" width="100%" height="100%"> 



