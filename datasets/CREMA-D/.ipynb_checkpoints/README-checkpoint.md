### How to install the CREMA-D dataset

- (See https://github.com/CheyneyComputerScience/CREMA-D)

1. Install git

~~~
sudo apt-get install git-all
~~~

2. Install git-lfs

- (See http://arfc.github.io/manual/guides/git-lfs)
~~~
git lfs install
~~~

3. Download CREMA-D into the datasets/CREMA-D/CREMA-D folder
~~~
git clone https://github.com/CheyneyComputerScience/CREMA-D.git
~~~

4. Install mmpeg and moviepy
~~~
sudo apt update
sudo apt upgrade
sudo apt install ffmpeg
sudo apt upgrade ffmpeg
pip install moviepy
~~~

- The instalation of moviepy on 2022/10 was buggy so the following command was used instead
~~~
pip install moviepy==2.0.0.dev2
~~~