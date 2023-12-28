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

Now we have confirmed that the codebase works in our in-house runtime, and we will soon add the usage on other platforms.

[comment]: <> (- First, install the dependencies in [CM-VQVAE/requirements.txt]CM-VQVAE/requirements.txt)

[comment]: <> (~~~)
[comment]: <> (pip3 install -U -r CM-VQVAE/requirements.txt)
[comment]: <> (~~~)

[comment]: <> (- Then, manually install the following packages:)

[comment]: <> (~~~)
[comment]: <> (pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 torchaudio==0.10.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html)

[comment]: <> (sudo apt-get install libsndfile1-dev)

[comment]: <> (sudo apt-get install libgl1)

[comment]: <> (sudo apt-get install ffmpeg libsm6 libxext6  -y)
[comment]: <> (~~~)

[comment]: <> (- Finally, go to [CM-VQVAE/CREMA-D]CM-VQVAE/CREMA-D and run)

[comment]: <> (~~~)
[comment]: <> (python main.py)
[comment]: <> (~~~)

[comment]: <> (- A manual of the code and functions can be found in [CM-VQVAE/CREMA-D/README.md]CM-VQVAE/CREMA-D/README.md)
