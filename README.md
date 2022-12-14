# Face Recognition

## Dependencies

- tensorflow >= r1.5
- opencv-python 3.x
- python 3.x
- scipy
- sklearn
- numpy
- mxnet
- pickle

## Prepare dataset

1. choose one of the following links to download dataset which is provide by insightface. (Special Recommend MS1M-refine-v2)
* [MS1M-refine-v2@BaiduDrive](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ), [MS1M-refine-v2@GoogleDrive](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
* [Refined-MS1M@BaiduDrive](https://pan.baidu.com/s/1nxmSCch), [Refined-MS1M@GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view)
* [VGGFace2@BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [VGGFace2@GoogleDrive](https://www.dropbox.com/s/m9pm1it7vsw3gj0/faces_vgg2_112x112.zip?dl=0)
* [Insightface Dataset Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
2. move dataset to `${MobileFaceNet_TF_ROOT}/datasets`.
3. run `${MobileFaceNet_TF_ROOT}/utils/data_process.py`.

## Training

1. refined super parameters by yourself special project.
2. run script
`${MobileFaceNet_TF_ROOT}/train_nets.py`
3. have a snapshot result at `${MobileFaceNet_TF_ROOT}/output`.

## Note
Create folder embeddings to contain embeddings
## Create embeddings  
python3 create_ctf.py -n name
## Recognize face
python3 recog_ctf.py

## References

1. [MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF)
2. [CenterFace](https://github.com/Star-Clouds/CenterFace)
3. [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)
4. [CenterFace: Joint Face Detection and Alignment Using Face as Point](https://arxiv.org/abs/1911.03599)



