Dual Path Denoising Network for Real Photographic Noise
=======================================================
Yeong Il Jang, Yoonsik Kim, and Nam Ik Cho

[[Paper](https://ieeexplore.ieee.org/document/9098102)]


Test code
----

### Environments

-	Windows10 / Ubuntu 16.04
-	Tensorflow1.12
-	Python 3.6

### Models
Pretrained models for real noise and AWGN can be downloaded from followed link:

[[Models](https://drive.google.com/drive/folders/1ZJxcgimNXd5T_dNvdoPzYYWRj1hF6cHc?usp=sharing)]

### Test

```
python test.py 
--imagepath : Path of test images
--savepath : Path of denoised images [default : './results/'] 
--model : Model checkpoint [real / AWGN] [default : './models/DPDN']
--add_noise : Add AWGN to test image [default : False]
--sigma : Standard deviation of AWGN (in [0 to 255] scale) [default : 25]
```

### Examples
To denoise real noisy images,
```
python test.py --imagepath ./RNI15/ --savepath ./RNIdenoised/ --model ./models/DPDN 
```

To denoise synthetic noisy images (AWGN),
```
python test.py --imagepath ./CBSD68/ --savepath ./CBSDdenoised/ --model ./models/AWGN --add_noise --simga 25
```


Experimental Results
--------------------

The denoised results for DPDN can be downloaded from followed links:

[DND](https://drive.google.com/file/d/14jQQNE9szQmS-GvnYkvAd5XXrfzMYORA/view?usp=sharing) [SIDDValidation](https://drive.google.com/file/d/1qQVAUyLM2AzinAz2XAEjJ8EOwLBJI6i3/view?usp=sharing) [RNI15](https://drive.google.com/file/d/1ab5gzMhVUdl7SVO2UPM-6yXKQAZOHhyE/view?usp=sharing)

### Visualized results

<p align="center"><img src = "/figs/DND1.PNG" width="900">
<p align="center"><img src = "/figs/SIDD.PNG" width="900">
<p align="center"><img src = "/figs/RNI.PNG" width="900">

Citation
--------
If you use the work released here for your research, please cite this paper:

```
@ARTICLE{9098102,
author={Y. I. {Jang} and Y. {Kim} and N. I. {Cho}},
journal={IEEE Signal Processing Letters},
title={Dual Path Denoising Network for Real Photographic Noise},
year={2020},
volume={27},
number={},
pages={860-864},}
```
