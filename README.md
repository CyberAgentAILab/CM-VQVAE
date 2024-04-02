# CM-VQVAE

This is the research code for the WACV2024 paper "Complementary-Contradictory Feature Regularization against Multimodal Overfitting".

```bibtex
@inproceedings{tejerodepablos2024complementary,
    title     = {Complementary-contradictory feature regularization against multimodal overfitting},
    author    = {Tejero-de-Pablos, Antonio},
    booktitle = {In Proc. Winter Conference on Applications of Computer Vision},
    pages     = {5679-5688},
    year={2024}
}
```

We propose a method to mitigate multimodal overfitting in classification tasks. For example, in an emotion recognition task, RGB video frames and audio signals are used to predict the actor's emotion. In the vanilla classification setting, multimodal features (fusing the visual modality with audio) performs worse than the visual-only unimodal features. By learning a masking operation on the multimodal features, obtrusive information (contradictory) is removed, and only the essential information (complementary) is used for classification.

Here we provide the implementation ready-to-run for the emotion recognition dataset CREMA-D (see the manual on datasets/CREMA-D/README.md for obtaining the dataset). Running the file [CM-VQVAE/CREMA-D/main.py](CM-VQVAE/CREMA-D/main.py) as below provides the experimental results for training/validation and test.

## How to run (python ver. 3.8.6)

[comment]: <> (Now we have confirmed that the codebase works in our in-house runtime, and we will soon add the usage on other platforms.)

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

- A manual of the code and functions can be found in [CM-VQVAE/CREMA-D/README.md](CM-VQVAE/CREMA-D/README.md).
