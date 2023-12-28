# CM-VQVAE

This is the research code for the WACV2024 paper "Complementary-Contradictory Feature Regularization against Multimodal Overfitting".

```bibtex
@inproceedings{tejerodepablos2024complementary,
    title     = {Complementary-contradictory feature regularization against multimodal overfitting},
    author    = {Tejero-de-Pablos, Antonio},
    booktitle = {In Proc. Winter Conference on Applications of Computer Vision},
    pages     = {1--10},
    year={2024}
}
```

Here we provide the implementation ready-to-run for the emotion recognition dataset CREMA-D (see the manual on datasets/CREMA-D/README.md for obtaining the dataset)

## How to run (python ver. 3.8.6)

- First, install the dependencies in [CM-VQVAE/requirements.txt](CM-VQVAE/requirements.txt)

~~~
pip3 install -U -r CM-VQVAE/requirements.txt
~~~

- Then, manually install the following packages:

~~~
pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 torchaudio==0.10.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html

sudo apt-get install libsndfile1-dev

sudo apt-get install libgl1

sudo apt-get install ffmpeg libsm6 libxext6  -y
~~~

- Finally, go to [CM-VQVAE/CREMA-D](CM-VQVAE/CREMA-D) and run

~~~
python main.py
~~~

- A manual of the code and functions can be found in [CM-VQVAE/CREMA-D/README.md](CM-VQVAE/CREMA-D/README.md)
