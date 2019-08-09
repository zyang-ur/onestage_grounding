# One-Stage Visual Grounding
[A Fast and Accurate One-Stage Approach to Visual Grounding](https://arxiv.org/)

by [Zhengyuan Yang](http://cs.rochester.edu/u/zyang39/), [Boqing Gong](http://boqinggong.info/), [Liwei Wang](http://www.deepcv.net/), Wenbing Huang, Dong Yu, and [Jiebo Luo](http://cs.rochester.edu/u/jluo)

IEEE International Conference on Computer Vision (ICCV), 2019


### Introduction
We propose a simple, fast, and accurate one-stage approach 
to visual grounding. For more details, please refer to our
[paper](https://arxiv.org/).

<!-- ![alt text](http://cs.rochester.edu/u/zyang39/VG_ICCV19.jpg 
"Framework") -->
<p align="center">
  <img src="http://cs.rochester.edu/u/zyang39/VG_ICCV19.jpg" width="75%"/>
</p>

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
    git clone https://github.com/zyang-ur/onestage_grounding.git
    ```

2. Prepare the submodules and associated data

* RefCOCO & ReferItGame Dataset: place the data or the soft link of dataset folder under ``./ln_data/``. We follow dataset structure [DMS](https://github.com/BCV-Uniandes/DMS). To accomplish this, the ``download_dataset.sh`` [bash script](https://github.com/BCV-Uniandes/DMS/blob/master/download_data.sh) from DMS can be used.
    ```bash
    bash download_data --path ./ln_data
    ```

* Flickr30K Entities Dataset: place the data or the soft link of dataset folder under ``./ln_data/``. The formated Flickr data is availble at [[Gdrive]](https://drive.google.com/open?id=1A1iWUWgRg7wV5qwOP_QVujOO4B8U-UYB), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/Eqgejwkq-hZIjCkhrgWbdIkB_yi3K4uqQyRCwf9CSe_zpQ?e=dtu8qF).

* Data index: download the generated index files and place them in the ``./data`` folder. Availble at [[Gdrive]](https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/Epw5WQ_mJ-tOlAbK5LxsnrsBElWwvNdU7aus0UIzWtwgKQ?e=XHQm7F).

* Model weights: download the pretrained model of [Yolov3](https://pjreddie.com/media/files/yolov3.weights) and place the file in ``./saved_models``. More pretrained models are availble in the performance table [[Gdrive]](https://drive.google.com/open?id=1-DXvhEbWQtVWAUT_-G19zlz-0Ekcj5d7), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/ErrXDnw1igFGghwbH5daoKwBX4vtE_erXbOo1JGnraCE4Q?e=tQUCk7) and should also be placed in ``./saved_models``.


### Training
3. Train the model, run the code under main folder. 
Using flag ``--lstm`` to access lstm encoder, Bert is used as the default. 
Using flag ``--light`` to access the light model.

    ```
    python train_yolo.py --data_root ./ln_data/ --dataset referit \
      --gpu gpu_id --batch_size 32 --resume saved_models/lstm_referit_model.pth.tar \
      --lr 1e-4 --nb_epoch 100 --lstm
    ```

4. Evaluate the model, run the code under main folder. 
Using flag ``--test`` to access test mode.

    ```
    python train_yolo.py --data_root ./ln_data/ --dataset referit \
      --gpu gpu_id --resume saved_models/lstm_referit_model.pth.tar \
      --lstm --test
    ```

5. Visulizations. Flag ``--save_plot`` will save visulizations.


## Performance and Pre-trained Models
Please check the detailed experiment settings in our [paper](https://arxiv.org/).
<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Ours-LSTM</th>
            <th>Performance (Accu@0.5)</th>
            <th>Ours-Bert</th>
            <th>Performance (Accu@0.5)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ReferItGame</td>
            <td><a href="https://drive.google.com/open?id=1-DXvhEbWQtVWAUT_-G19zlz-0Ekcj5d7">Gdrive</a></td>
            <td>58.76</td>
            <td><a href="https://drive.google.com/open?id=1-DXvhEbWQtVWAUT_-G19zlz-0Ekcj5d7">Gdrive</a></td>
            <td>59.58</td>
        </tr>
        <tr>
            <td>Flickr30K Entities</td>
            <td><a href="https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/ErrXDnw1igFGghwbH5daoKwBX4vtE_erXbOo1JGnraCE4Q?e=tQUCk7">One Drive</a></td>
            <td>67.62</td>
            <td><a href="https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/ErrXDnw1igFGghwbH5daoKwBX4vtE_erXbOo1JGnraCE4Q?e=tQUCk7">One Drive</a></td>
            <td>68.69</td>
        </tr>
        <tr>
            <td rowspan=3>RefCOCO</td>
            <td rowspan=3><!-- <a href="https://drive.google.com/open?id=1-DXvhEbWQtVWAUT_-G19zlz-0Ekcj5d7">Weights</a></td> -->
            <td>val: 73.66</td>
            <td rowspan=3><!-- <a href="https://drive.google.com/open?id=1-DXvhEbWQtVWAUT_-G19zlz-0Ekcj5d7">Weights</a></td> -->
            <td>val: 72.05</td>
        </tr>
        <tr>
            <td>testA: 75.78</td>
            <td>testA: 74.81</td>
        </tr>
        <tr>
            <td>testB: 71.32</td>
            <td>testB: 67.59</td>
        </tr>
    </tbody>
</table>


### Credits
Part of the code or models are from 
[DMS](https://github.com/BCV-Uniandes/DMS),
[MAttNet](https://github.com/lichengunc/MAttNet),
[Yolov3](https://pjreddie.com/darknet/yolo/) and
[Pytorch-yolov3](https://github.com/eriklindernoren/PyTorch-YOLOv3).
