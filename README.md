<!-- ## Introduction
one-stage visual grounding; minial version
## preparation

* Python 3.5
* Pytorch 0.4 or higher
* Pytorch-Bert https://github.com/huggingface/pytorch-pretrained-BERT
* Yolov3.weights https://pjreddie.com/media/files/yolov3.weights
* Datasets


## Training
1. Dataset: place the soft link of dataset folder in code/../ln_data/DMS/...
	We follow dataset structure from https://github.com/BCV-Uniandes/DMS
2. train_yolo.py is the main training and validation file; check darknet.py and textcam_yolo.py
	for models. referit_loader.py is the used dataloader

This repo is partly built on the YOLOv3 implementation (https://github.com/eriklindernoren/PyTorch-YOLOv3) and the data loader implemented by DMS (https://github.com/BCV-Uniandes/DMS). -->


# A Fast and Accurate One-Stage Approach to Visual Grounding
by [Zhengyuan Yang](http://cs.rochester.edu/u/zyang39/), Boqing Gong, Liwei Wang, Wenbing Huang, Dong Yu, and [Jiebo Luo](http://cs.rochester.edu/u/jluo)

### Introduction
We propose a simple, fast, and accurate one-stage approach 
to visual grounding. For more details, please refer to our
[paper](https://arxiv.org/).

![alt text](http://cs.rochester.edu/u/zyang39/VG_ICCV19.jpg 
"Framework")

### Citation

    @inproceedings{yang2019fast,
      title={A Fast and Accurate One-Stage Approach to Visual Grounding},
      author={Yang, Zhengyuan and Gong, Boqing and Wang, Liwei and Huang
        , Wenbing and Yu, Dong and Luo, Jiebo},
      booktitle={ICCV},
      year={2019}
    }

### Prerequisites

* Python 3.5
* Pytorch 0.4.1 (1.0 or higher versions to be tested)
* Others ([Pytorch-Bert](https://github.com/huggingface/pytorch-pretrained-BERT))

## Installation

1. Clone the repository

```
git clone --recursive https://github.com/zyang-ur/onestage_grounding.git
```

2. Prepare the submodules and associated data

* RefCOCO & ReferItGame Dataset: place the soft link of dataset folder in ./ln_data/DMS/. We follow dataset structure [DMS](from https://github.com/BCV-Uniandes/DMS). To accomplish this, the ``download_dataset.sh`` [bash script](https://github.com/BCV-Uniandes/DMS/blob/master/download_data.sh) from DMS can be used.
```bash
bash download_data --path $PATH_TO_STORE_THE_DATASETS
```

* Flickr30K Entities Dataset: place the soft link of dataset folder in ./ln_data/DMS/. The formated Flickr data is availble in the following link.

* Data index: download the generated index files and place them in the data folder. Availble at [Gdrive](https://drive.google.com/open?id=1i9fjhZ3cmn5YOxlacGMpcxWmrNnRNU4B). A copy at OneDrive is also availble.

* Model weights: download the pretrained model of [Yolov3](https://pjreddie.com/media/files/yolov3.weights) and place the file in ./code/saved_models. More pretrained models are availble in the performance table and should also be placed in ./code/saved_models.


### Training
Train the model, run the code under folder ./code. 
Using flag --lstm to access lstm encoder, Bert is used as the default. 
Using flag --light to access the light model.
    ```
    python train_yolo.py --data_root ../ln_data/DMS/ --dataset referit \
      --gpu gpu_id --batch_size 32 --resume saved_models/model.pth.tar \
      --lr 1e-4 --nb_epoch 100 --lstm
    ```

Evaluate the model, run the code under folder ./code. 
Using flag --test to access test mode.
    ```
    python train_yolo.py --data_root ../ln_data/DMS/ --dataset referit \
      --gpu gpu_id --resume saved_models/model.pth.tar \
      --lstm --test
    ```

Visulizations. Flag --save_plot will save visulizations.

9. Evaluate the model on test set. Suppose the best validation checkpoint
is 20000.
    ```
    python test_model.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --data_dir ~/dataset/mscoco/all_images --job_dir saving/model.ckpt-20000
    ```


## Performance and Pre-trained Models
Please check the detailed experiment settings in our [paper](https://arxiv.org/).
<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Ours-LSTM</th>
            <th>Performance (mIoU)</th>
            <th>Ours-Bert</th>
            <th>Performance (mIoU)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ReferItGame</td>
            <td><a href="http://cs.rochester.edu/u/zyang39/">Weights</a></td>
            <td>58.76</td>
            <td><a href="http://cs.rochester.edu/u/zyang39/">Weights</a></td>
            <td>59.58</td>
        </tr>
        <tr>
            <td>Flickr30K Entities</td>
            <td><a href="http://cs.rochester.edu/u/zyang39/">Weights</a></td>
            <td>-</td>
            <td><a href="http://cs.rochester.edu/u/zyang39/">Weights</a></td>
            <td>68.91</td>
        </tr>
        <tr>
            <td rowspan=3>UNC</td>
            <td rowspan=3><a href="http://cs.rochester.edu/u/zyang39/">Weights</a></td>
            <td>val: 73.66</td>
        </tr>
        <tr>
            <td>testA: 75.78</td>
        </tr>
        <tr>
            <td>testB: 71.32</td>
        </tr>
            <td rowspan=3><a href="http://cs.rochester.edu/u/zyang39/">Weights</a></td>
            <td>val: 72.44</td>
        </tr>
        <tr>
            <td>testA: 75.13</td>
        </tr>
        <tr>
            <td>testB: 68.05</td>
        </tr>
    </tbody>
</table>


### Credits
Part of the code or models are from 
[DMS](https://github.com/BCV-Uniandes/DMS),
[MAttNet](https://github.com/lichengunc/MAttNet),
[Yolov3](https://pjreddie.com/darknet/yolo/) and
[Pytorch-yolov3](https://github.com/eriklindernoren/PyTorch-YOLOv3).