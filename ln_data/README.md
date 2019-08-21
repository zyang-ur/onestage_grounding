# Data Folder
* RefCOCO & ReferItGame Dataset: place the soft link of dataset folder under the current folder. We follow dataset structure [DMS](https://github.com/BCV-Uniandes/DMS). To accomplish this, the ``download_dataset.sh`` [bash script](https://github.com/BCV-Uniandes/DMS/blob/master/download_data.sh) from DMS can be used.
    ```bash
    bash download_data --path .
    ```

<!-- * Flickr30K Entities Dataset: place the data or the soft link of dataset folder under the current folder. The formated Flickr data is availble at [[Gdrive]](https://drive.google.com/open?id=1A1iWUWgRg7wV5qwOP_QVujOO4B8U-UYB), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/Eqgejwkq-hZIjCkhrgWbdIkB_yi3K4uqQyRCwf9CSe_zpQ?e=dtu8qF).
    ```
    tar xf Flickr30k.tar
    ``` -->
* Flickr30K Entities Dataset: please download the images for the dataset on the website for the [Flickr30K Entities Dataset](http://bryanplummer.com/Flickr30kEntities/) and the original [Flickr30k Dataset](http://shannon.cs.illinois.edu/DenotationGraph/). Images should be placed under ``./Flickr30k/flickr30k_images``.

* Data index: download the generated index files and place them in the ``../data`` folder. Availble at [[Gdrive]](https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/Epw5WQ_mJ-tOlAbK5LxsnrsBElWwvNdU7aus0UIzWtwgKQ?e=XHQm7F).
    ```
    cd ..
    rm -r data
    tar xf data.tar
    ```