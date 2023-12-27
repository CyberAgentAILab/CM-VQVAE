# Source code

## main.py

This python program is called to run the experiments for "complementary/contradictory multimodal features" on the CREMA-D dataset. The configuration parameters can be selected by modifying the first lines of code (number of epochs, etc). In particular, for training:

- message: String to be recorded at the beginning of the log. Its purpose is to leave a brief description of the intentions of the experiment, so it is easier to distinguish its results from the others.

- timestamp: If 'None', the training will be executed from scratch. If a value of the type '20220413-094829' is introduced, the corresponding learned model will be read from the results folder, and the evaluation will be executed (training is skipped).

- dataset: Please use 'CREMAD'.
    - CREMAD: Emotion recognition dataset with actors of different genders and nationalities.
This variable is also input to the model to initialize the architecture. Depending on the dataset, the following variables will vary accordingly.
    - Dataset for the dataloader
    - Names of the labels
    - Loss function (criterion)

- pretrained: This flag is used when using the RML dataset, as it does not have enough samples to successfully train our model. When set to True, the program will read a pretrained model (i.e., from CREMAD) and replace the VQVAE modules from the RML model (encoder, decoder, codebook, etc). Also, since the CREMAD model uses spectrograms of size 512x512 for the audio modality, this flag makes the dataset of RML spectrograms have the same size (see datasets.py).

- mode: It can take three possible values depending on the modalities that will be used for the experiment.
    - 'multimodal': Both modalities will be used to train/test the model
    - 'image-only': Only the image modality data will be used
    - 'audio-only': Only the audio modality data will be used

- base_type: Configures whether to use the proposed method, or a baseline that lacks certain module(s). It is used for ablation study purposes.
    - 0: No baseline (normal)
    - 1: No reconstruction and no masking (2+3)
    - 2: No masking in TaskSolver
    - 3: No reconstruction

- data_path: Folder where the configuration necessary for data loading is stored. The folder contains configs for each split (train, val, test) and modality (see the details on the corresponding sections below). It can take the following values:
    - 'data_cremad'
    - 'data_rml'
    - 'data_rml_by_user'

- balanced: This is deprecated, and the value should be False. If True, the balanced version of the CREMA dataset will be used (only four classes are used, and the number of samples per-class is comparable).

- num_epoch: Training epochs. Recommended: 35 (10 if 'data_rml_by_user' due to overfitting)

- batch_size: Batch size. Recommended: 128

- shuffle ... threshold: This set of hyperparameters control the learning and the architecture. They were determined empirically, so they should not be modified in principle.

- regularization: Add a regularization term to the loss to penalize masks with many '1' values

- modify_modal: In the RML dataset, adds contradictory features (i.e., skin color) to the given task (emotion recognition)

Compared to the model used for the Digits dataset, the sizes for the model architecture are larger, given the higher complexity of the task (emotion classification) and the dimensionality of the modalities. Also, the VAE architecture selection is not implemented, since experiments with the comparatively simpler Digits dataset showed that VAE is less suitable to this scenario than VQVAE.

Depending on the dataset used, the corresponding model architecture, data loader and label names are defined.

On each training epoch, the emotion recognition modalities (human face and its respective audio) are input to the model, which in turn returns their respective reconstructions, predicted emotion label, VQVAE losses, VQVAE perplexity and shared loss regularization term. Both the reconstruction and the classification losses are calculated via the mean squared error, and are added along with the VQVAE losses for the final loss.

The emotion recognition datasets contain three splits: training, validation and test. After each learning epoch with the training split, the model is evaluated in the validation split (no weight update) to check for overfitting and other training phenomena. The model with the lowest loss during validation is saved for use in the evaluation (test) phase.

Once the training is over, a timestamp with the current date and time is generated and used to name the results folder. Then, the model and log are saved along with some training statistics, namely, the reconstruction loss, the perplexity, and the codebooks produced by each VQVAE.

Then, for evaluation, the predicted class with the highest probability and the label are compared sample by sample, and the class wise accuracy is calculated.

Finally, the learned private and shared spaces of each modality, as well as the modal complementarity are displayed. Then, the last batch of evaluation samples (originals and reconstructed) are saved.

To summarize, executing the file results on a log, a set of result files (see the results folder explanation below) and a printed output (that is logged as well).

### read_variances.py

This function reads the variances of the visual/audio modalities in the training data split. The variance values were calculated by 'read_cremad.py'.


## datasets.py

This python file implements a Dataset class that can be used by a data loader for the CREMA-D dataset.

### EmoVoxCelebDataset

It requires the files in the 'data' folder generated by 'read_emo.py'. When initialized, the file paths for the image frames and audio files of the given split (train, val, test) is loaded, along with the logits label data, and an index that indicates which label belongs to which file. Then, if only a subset of the data is used, the aforementioned data is subsampled.

When reading a sample from the dataset, the corresponding image frame, audio and label for the given index are read and returned.
- Image modality:
    - In the current version, a single frame is chosen randomly from the indexed video and read as our image modality.
    - The image is resized to 128x128 to facilitate processing, and converted to Torch float type.
- Audio modality:
    - The audio file is read (1D signal), and transformed to a mel-spectrogram (2D signal)
    - The hardcoded parameters used to create the mel-spectrogram are the recommended for processing human voice.
    - The mel-spectrogram scale is changed from power to decibels (dB)
    - The tensor is resized to 512x512 and converted to Torch float type. Since the length of the audio file varies, so does the size of the mel-spectrogram. We take a compromised intermediate value of 512x512, which makes some samples sound faster/slower than the original audio.

### RMLDataset

In this dataset, visual and audio modalities are obtained from the same video file, stored in the 'data' folder, along with the labels. When initialized, these files are loaded from their respective file paths and the given split (train, val, test).

When reading a sample from the dataset, the corresponding image frame, audio and label for the given index are read and returned. When reading a video, the visual modality, audio modality and sample_rate are obtained.
- Image modality:
    - In the current version, a single frame is chosen randomly from the central part of the video (beginning and end do not contain emotions).
    - Then, the central region of the video contained the face size is cropped and resized to 128x128.
- Audio modality:
    - The audio signal is cropped to its central 3 seconds (beginning and end does not contain emotions) and resampled to 16kHz.
    - Then a mel-spectrogram of 128x128 is calculated, and its scale is changed from power to decibels (dB)
    - If the duration of the spectrogram is not 128, it is padded with zeros.

Finally, tensors are converted to Torch float type. In the case of using the RML dataset to fine-tune a pretrained model, the audio modality is matched to that of EmoVoxCeleb:
- Audio modality:
    - The audio signal is not cropped.
    - The mel-spectrogram is 512, and resized to 512x512.

There is an option to add irrelevant features to the modalities (i.e., modified skin color to the images and modified pitch to the audio).

### RMLFeatDataset

Instead of the raw videos, this dataset reads the image and audio features previously extracted from the RML dataset, following the same configurations of the RML config. For the audio modality, since features from 24 layers of the feature extractor are obtained, indicating a layer ID is necessary (empirically, the 7th is the most effective).

Since the features extracted via 'pretrained_features_create' are obtained by previously preprocessing the videos (image, audio), this class only needs to reencapsulate them in a tensor. Labels are read as is, and returned along with the image and audio features.

### RMLProbeDataset

It adds "emotion contradictory" class labels (i.e., skin color, voice pitch, user) to the RMLDataset class.

### CREMADDataset

Same functionality as RMLDataset, but adapted to CREMA-D. Since videos are in FLV format, moviepy is used for reading, but it is slow (we recommend resave the image frames in individual files and read via torchvision). Since CREMA-D has more abundant and varied samples, the following funcitonalities were not implemented: pretraining, modified modalities ("emotion irrelevant" modifications), audio subsampling.

Besides the emotion class label, the CREMA-D dataset provides other type of information:
- Actor ID that appears in the sample
- Sentence ID that is said by the actor
- Level of emotional content (low, neutral, high, unknown)
- Age of the actor
- Gender of the actor
- Race of the actor (African American, Caucassian, Asian, unknown)
- Ethnicity of the actor (Spanish or non-Spanish)
The labels above are extracted, but only the emotion, race, gender and sentence labels are returned.


## model.py

This python file contains the proposed network architecture and a separate model variation for the ablation study experiments (used in 'main_separate.py'). The main architecture is composed of a VQVAE backbone for each modality and a task-solver module. Each VQVAE backbone is in turn composed of an encoder, a vector-quantizer and a decoder. Further details on the implementation of these modules are provided below.

The code for VQVAE is based on: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb

### VectorQuantizer

This class instantiates a VQVAE model. Such model first reshapes the input encoder feature map into feature vectors that match the codebook's code size. Then, for each reshaped feature vector, the distance with each code of the codebook is calculated, and the feature vector is replaced with the closest code (quantization). Then, the VQVAE loss is calculated based on the distance between the input feature vectors and the quantized ones.

Finally, a VQVAE perplexity value (codebook usage) is calculated.

### VectorQuantizerEMA

This class is equivalent to the 'VectorQuantizer', but its training is more efficient, thanks to the EMA algorithm to update the codebook.

### Residual

Implements a residual layer/block of a ResNet backbone.

### ResidualStack

This class module instantiates a ResNet backbone composed of a concatenation of 'num_residual_layers' of Residual classes.

### EncoderImage

This class module is composed of an initial convolution layer, and a ResNet backbone. The network size is adjusted to the image modality.

### EncoderAudio

Same as the 'EncoderImage' class, but its size is adjusted to the audio modality.

### DecoderImage

This class module is composed of a convolution layer, a ResNet backbone, and two transposed convolutions. The network size is adjusted to the image modality.

### DecoderAudio

Same as the 'DecoderImage' class, but its size is adjusted to the audio modality.

### TaskSolver

This class implements the module that learns a private-shared disentangled latent space. It contains a learnable mask variable whose elements are in the range [0,1] ([-1,-1] for the ternary case). This mask is binarized into {0,1} values ({-1,0,-1} for the ternary case) according to a threshold variable (except for the attention case, which uses continous values). Each of these values is intended to mask one of the feature map units conforming the concatenated output of the modal-wise VectorQuantizer. If the mode hyperparameter is not 'multimodal', only one of the modalities will be used for classification (next paragraph).

The masked encodings are then input to two convolutional layers and two fully-connected layers, and the result is the prediction logits of the emotion classes (6 probabilities). This way, only the information from the non-masked elements is used to solve the given task.

More info on the implementation of the learnable mask can be found here and the links in the code: https://github.com/arunmallya/piggyback/blob/master/src/modnets/layers.py

#### get_private_shared_ratio

This function accesses the mask of the task-solver module, and returns the private-shared ratios of the latent space of each modality, along with their complementarity measure. These ratios are calculated by counting the 0's and 1's of the learned masked, with resepct to the total mask size. Then, the complementarity is calculated by dividing the smaller shared ratio by the bigger shared ratio.

#### get_mask

This function returns the mask variable from the TaskSolver. It is called when passing the learned parameters from a 'Model' class to a 'Model_Probe'.

#### reset_mask

This function is used for transfer learning from a pretrained CM-VQVAE model. It allows resetting the learned mask to its initial values, in order to compare different configurations of knowledge transfer between models.

### Binarizer, Binarizer_Inverse, Ternarizer and Randomizer

These classes get the learned mask, and binarize (or ternarize) its values by comparing them to a given threshold. This step is carried out previously to masking.
- *_Inverse versions of the functions allow working with private spaces in other parts of the model. They assign 0s to the shared features and 1s to the private ones.

The Randomizer class implements a baseline in which the concatenated modal-wise feature units are masked randomly, instead of learning the binarized weights. This is to prove that our masking method is not a simple dropout technique.

### Model

This class is used to create an instance of our proposed architecture. It consists of a VQVAE module (encoder, pre-VQ convolution, VQ, decoder) per each modality. It gets the image and audio modalities as input, and returns the VQ losses, reconstructed images, and perplexity values for each modality, as well as the prediction from the task solver and a shared-loss regularization term.

### Model_VQVAE

This class encapsulates the VQVAE's module of the proposed method in order to train it separately from the task solver. Thus, it takes the image and audio modalities as inputs, and outputs their VQ losses, reconstructions, and perplexity values. Additionally, the quantized encodings of the VQ class are returned, to be subsequently used by the separate Model_Solver class.

### Model_Solver

This class encapsulates the task-solver module of the proposed method, in order to train it separately from the VQVAE's. It takes the quantized encodings of the modalities as inputs, and returns the class predictions vector and a shared-loss regularization term.

### Model_Feat

This class adapts the 'Model' class to work with pre-extracted features. Modalities do not need an encoder and decoder, as the reconstructions are the features themselves, encoded via a VQVAE. Thus, only the VQVAE bottleneck, and a dedicated TaskSolver (see class below) are needed. The pretrained features are resized to the same size as the general 'Model' bottleneck.

### TaskSolver_Feat

This is the 'TaskSolver' class above adapted to the pre-extracted features of 'Model_feat'. Instead of convolutions, here feature projection into a lower dimensionality is performed via linear layers. Then, image and video features from different time frames are concatenated and modeled via an LSTM. Since the features size is adapted to the bottleneck of the original 'Model', the learnable mask can be applied as is.
As the rest of the network, batch normalization is applied.

#### subsample_features

This function takes an audio features vector of 64 time frames, and subsamples it to 16.

#### get_private_shared_ratio

This function is the same as the 'get_private_shared_ratio' above.

### Model_Baseline

In order to check the influence of our proposed method in the performance of the multimodal pipeline, we define a model that simply solves the task using the given modalities without doing any kind of disentanglement. More specifically:
- We keep: Encoders, pre-VQVAE convs, classifier (task solver)
- We remove: VQVAEs (baseline 1 and 3), decoders (baseline 1 and 3), masks (baseline 1 and 2)

### TaskSolver_Baseline

This class implements the task solver module of the proposed method, but in order to be used specifically for the Model_Baseline class. It is basically the same module as the TaskSolver class, but without the mask processing and the get_private_shared_ratio function.

### Model_Probe

This class implements an encoder-VQVAE-decoder pipeline as in the 'Model' class, changing the 'TaskSolver' for a 'Masker' module that removes private/shared codes before the reconstruction. Thus, this module does not have a classification prediction output, but only the decoded reconstructions of the input using only part of its latent codes. The codes used are indicated by the 'invert' variable (see 'Masker' class).

### Masker

This class implements the masker module of the probing model. It is based on the 'TaskSolver' class, but leaving only the masking functionality for each modality. The input parameter 'invert' indicates which codes will be masked:
- If False, the private codes will be masked and only the shared codes will be returned (the Binarizer function is called).
- If True, the shared codes will be masked and only the private codes will be returned (the Binarizer_Inverse function is called).

#### mix

After calculating (masking) the private/shared features of the input samples, this function mixes the shared features of a sample with the private features of the previous sample, for later reconstruction. This function returns the mixed features for each modality.

#### get_private_shared_ratio

Same function as the 'TaskSolver'.

#### set_mask

This function sets the mask variable to the mask that receives as input. It is called when copying the parameters from an already trained model to the probing model.

### Model_Comparison

This class implements a model based on two ResNet18 (one for each modality) as the backbone architecture defined in the paper: "Balanced Multimodal Learning via On-the-fly Gradient Modulation". It facilitates conducting an experiment to compare the performance of our baseline 1 (no reconstruction nor masking) and this backbone model, in order to show that there is no significant difference. The inputs are adapted to the size of each modality, and the output is concatenated to a linear classifier. Along with the predictions, dummy values for the modality reconstructions and VQVAE losses are returned.

### Dummy_Classifier

This class is necessary to return dummy values for the private-shared space sizes of the Model_Comparison model (these values are not actually used).


## read_cremad.py

This python program reads the files of the CREMA-D dataset, and creates some configuration files that are used by the pytorch dataset class in order to load video and labels for the train/validation/test splits.

First, the origin and destination folders are defined, as well as the number of samples per split. Then, the class names are defined for each label type. If the destination folder does not exist, it is created and configuration files are created for each split, and their class ratio printed.

When creating config files, first, the samples are read from the dataset folder (see 'read_samples' function below). Then, for each split in the dataset, a configuration file is created (see 'create_config' function below) and the class ratios are calculated (see 'print_class_ratio' function below).

Then, as some algorithms use the variance of the data in order to calculate the reconstruction loss, we also load the train split of the dataset to calculate this value. For this, we instantiate a dataset class of CREMA-D with the config files previously generated. We take all the training samples and read them at once. Finally, the max and variance values for each modality are printed out and saved in the 'data' folder:
- train_image_var.pkl: Contains the variance of the image modality
- train_audio_var.pkl: Contains the variance of the audio modality

NOTE: The number of samples on each split is based on the used in: "Balanced Multimodal Learning via On-the-fly Gradient Modulation".

### read_samples

This function reads two configuration CSV files to extract the following information:
- VideoDemographics.csv: Contains the age, gender, race and ethnicity of the actors.
- SentenceFilenames.csv: Contains the names of the video files, which are added to the dataset root path.

Then, the list of videos is randomly sorted.

### create_config

This function takes the path of the input videos and their labels, and creates a configuration file for the given split.

For each video sample, first, the following labels are extracted from the file name: actor ID, sentence ID, emotion class and emotion level. Then, the video demographics info read above is also stored as labels. In this process, the number of labels for each class is counted, to calculate the class ratios as below.

Finally, the paths of the video samples and their corresponding labels are stored.

NOTE: Since there are faulty files in the dataset (four to our knowledge), we can decide to omit them for the experiments.

### print_class_ratio

For each label type (emotion, race, etc.) and for each class in the label type, the ratio of each class is calculated and printed to a file.

### get_age_class

Converts the input integer to a discrete label indicated by the closest multiple of 10 (e.g., 46 --> '40s').


# Folders

## results

This folder contains the files resulting from training a model and its evaluation (a single execution of 'main.py' or 'main_separate.py'). These results are stored in a folder with the name of the timestamp when the model was learned (generated automatically). The timestamp folder contents are:
- log.txt: Contains the values of the config variables used to train the model (num epoch, etc.), and a copy of the printed output for training and evaluation.
- model.pt: Learned model ready to be read into a pytorch variable (for evaluation, etc.) generated by 'main.py'
- model_vae.pt: Learned VQVAE model ready to be read into a pytorch variable (generated by 'main_separate.py')
- train_class_error.pkl: Sequence of loss values for each epoch of the training of the proposed method (train split)
- val_class_error.pkl: Sequence of loss values for each epoch of the validation of the proposed method (val split)
- model_solver.pt:  Learned task-solver model ready to be read into a pytorch variable (generated by 'main_separate.py')
- train_res_perplexityN.pkl: Sequence of perplexity values obtained for each training iteration (batch) of the VQVAE of modality N
- val_res_perplexityN.pkl: Sequence of perplexity values obtained for each validation iteration (epoch) of the VQVAE of modality N
- train_res_recon_errorN.pkl: Sequence of reconstruction error values obtained for each training iteration (batch) of the VQVAE of modality N
- val_res_recon_errorN.pkl: Sequence of reconstruction error values obtained for each validation iteration (epoch) of the VQVAE of modality N
- valid_originalsN.pkl: Input samples of modality N during the last iteration of evaluation
- valid_reconstructionsN.pkl: Reconstructed samples of modality N during the last iteration of evaluation
- weightN.pkl: Learned codebook of the VQVAE of modality N


## data_cremad

Folder with the files generated in 'read_cremad.py' (see above). These are, for each split, the video samples list and their respective labels, as well as the class ratios (stats). In addition, the modal-wise variance values for the train data are also stored.
