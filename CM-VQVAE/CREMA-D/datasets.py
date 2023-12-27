from os.path import join, exists
from torchnet.dataset.dataset import Dataset

from moviepy.editor import VideoFileClip

from torchvision.io import read_image, read_video
from torchvision.transforms import Resize, ConvertImageDtype, ToTensor
from torchaudio import load as read_audio
from torchaudio.functional import resample
#from torchaudio.transforms import MelSpectrogram
#from torchaudio.functional import amplitude_to_DB
#from librosa import load as read_audio
from librosa.feature import melspectrogram
from librosa.feature.inverse import mel_to_audio
from librosa import power_to_db, db_to_power

from torch import float32
from torch.nn.functional import pad

from random import randint
import numpy as np
import pickle


# https://pytorch.org/docs/stable/data.html
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class EmoVoxCelebDataset(Dataset):
    def __init__(self, data_path, split, pretrained=False, subsample=None):

        with open(join(data_path, split + "_image.pkl"), "rb") as fp:  # Pickling
            self.image_path = pickle.load(fp)
        with open(join(data_path, split + "_audio.pkl"), "rb") as fp:  # Pickling
            self.audio_path = pickle.load(fp)
        with open(join(data_path, split + "_labels.pkl"), "rb") as fp:  # Pickling
            self.labels = pickle.load(fp)
        with open(join(data_path, split + "_index.pkl"), "rb") as fp:  # Pickling
            self.split_idx = pickle.load(fp)

        if subsample:
            samples = np.arange(0, len(self.split_idx), int(np.ceil(len(self.split_idx) / float(subsample))))
            self.image_path = [self.image_path[s] for s in samples]
            self.audio_path = [self.audio_path[s] for s in samples]
            self.split_idx = [self.split_idx[s] for s in samples]

        self.totensor = ToTensor()
        self.convert = ConvertImageDtype(float32)
        self.resize_image = Resize((128, 128)) # 128x128 is better, but we take a compromise with audio
        self.resize_audio = Resize((512, 512)) # 512x512 is better, but we take a compromise with image
        #self.mel_spectrogram = MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160)

    def __len__(self):
        return len(self.split_idx)

    def __getitem__(self, idx):
        path, n_images = self.image_path[idx]
        image_id = randint(0, n_images-1)
        image = read_image(join(path, "{}.jpg".format(str(image_id+1).zfill(6))))
        image = self.resize_image(image)#.float()
        image = self.convert(image)
        
        #waveform, sample_rate = read_audio(self.audio_path[idx], sr=None)
        waveform, sample_rate = read_audio(self.audio_path[idx])  # sample_rate = 16000
        audio = melspectrogram(y=waveform.numpy()[0], sr=sample_rate, n_fft=512, hop_length=160, win_length=400)
        #audio = self.mel_spectrogram(waveform) Use librosa instead of torchaudio
        audio = power_to_db(audio)
        #audio = amplitude_to_DB(audio, 10, 0, 1, 80) Use librosa instead of torchaudio
        audio = self.totensor(audio)
        audio = self.resize_audio(audio)#.float()
        audio = self.convert(audio)
        
        index = self.split_idx[idx]
        label = self.labels[index][image_id]

        return image, audio, label


class RMLDataset(Dataset):
    def __init__(self, data_path, split, pretrained, modify_modal):

        with open(join(data_path, split + "_videos.pkl"), "rb") as fp:  # Pickling
            self.video_path = pickle.load(fp)
        with open(join(data_path, split + "_labels.pkl"), "rb") as fp:  # Pickling
            self.labels = pickle.load(fp)
        self.pretrained = pretrained
        self.modify_modal = modify_modal

        self.crop_size = 190
        self.crop_sec = 2.5
        self.sub_sr = 32000
        self.totensor = ToTensor()
        self.convert = ConvertImageDtype(float32)
        self.resize_image = Resize((128, 128)) # 128x128 is better, but we take a compromise with audio
        self.resize_audio = Resize((512, 512)) # 512x512 is better, but we take a compromise with image
        #self.mel_spectrogram = MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frames, waveform, info = read_video(self.video_path[idx])
        
        if frames.shape[1] != 240:
            crop = 2 * self.crop_size
        else:
            crop = self.crop_size
        start_x = int(frames.shape[1]/2 - crop/2)
        start_y = int(frames.shape[2]/2 - crop/2)
        image_id = int(len(frames)/2 + randint(0, 45) - 22)  # Take a random central frame (beginning and end do not contain emotions)
        image = frames[image_id,start_x:start_x+crop,start_y:start_y+crop,:]  # Crop a central area of 128x128
        image = image.permute(2,0,1)
        image = self.resize_image(image)
        if False:
            _, image = mod_face(image)
        image = self.convert(image)
        
        sample_rate = int(info['audio_fps']*2)
        if not self.pretrained and waveform.shape[1]/sample_rate > self.crop_sec:
            start = int(waveform.shape[1]/2 - (sample_rate * self.crop_sec)/2)
            stop = int(start + (sample_rate * self.crop_sec))
            waveform = waveform[0, start:stop]
        else:
            waveform = waveform[0]
        subsampled = resample(waveform.type(float32), int(sample_rate), self.sub_sr)
        if False:
            _, subsampled = mod_voice(subsampled, int(sample_rate))
        
        if self.pretrained:
            audio = melspectrogram(y=subsampled.numpy().astype(float), sr=self.sub_sr, n_fft=512, hop_length=160, win_length=400)
        else:
            audio = melspectrogram(y=subsampled.numpy().astype(float), sr=self.sub_sr, n_fft=2048, win_length=512, hop_length=353, n_mels=256)
        #audio = self.mel_spectrogram(waveform) Use librosa instead of torchaudio
        audio = power_to_db(audio)
        #audio = amplitude_to_DB(audio, 10, 0, 1, 80) Use librosa instead of torchaudio
        if self.pretrained:
            audio = self.convert(self.resize_audio(self.totensor(audio)))
        else:
            crop = 256
            if audio.shape[1] < crop:
                filler = int((crop - audio.shape[1])/2)
                if np.pad(audio, ((0,0),(filler,filler)), "reflect").shape[1] != 256:
                    audio = np.pad(audio, ((0,0),(filler+1,filler)), "reflect")
                    problem = 1#DEBUG
                else:
                    audio = np.pad(audio, ((0,0),(filler,filler)), "reflect")
                    problem = 2#DEBUG
            elif audio.shape[1] > crop:
                remove = int((audio.shape[1] - crop)/2)
                if audio[:,remove:-remove].shape[1] != 256:
                    audio = audio[:,remove:-(remove+1)]
                    problem = 3#DEBUG
                else:
                    audio = audio[:,remove:-remove]
                    problem = 4#DEBUG
            if audio.shape[1] != crop:
                print("DEBUG {}: Different spectrogram sizes --> {}".format(problem, audio.shape[1]))
                exit()
            audio = self.convert(self.totensor(audio))
        
        label = self.labels[idx]
        
        return image, audio, label, -1, -1, -1


class CREMADDataset(Dataset):
    def __init__(self, data_path, split, pretrained, modify_modal):

        with open(join(data_path, split + "_videos.pkl"), "rb") as fp:  # Pickling
            self.video_path = pickle.load(fp)
        with open(join(data_path, split + "_labels.pkl"), "rb") as fp:  # Pickling
            self.labels = pickle.load(fp)

        self.sample_rate = 44100  # Obtained from the source data (see data_check.ipynb)
        self.fps = 30  # Obtained from the source data (see data_check.ipynb)
        self.crop_img = 280
        self.crop_spc = 256#224
        self.totensor = ToTensor()
        self.convert = ConvertImageDtype(float32)
        self.resize_image = Resize((128, 128))#Resize((224, 224))
        #self.mel_spectrogram = MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160)

    def __len__(self):
        return len(self.labels[2])

    def __getitem__(self, idx):
        #frames, waveform, info = read_video(self.video_path[idx])
        #DEBUGvideo = join('/home/jovyan/data/datasets/CREMA-D/CREMA-D/VideoFlash', self.video_path[idx].rsplit('/', 1)[-1])
        clip = VideoFileClip(join('..',self.video_path[idx]))
        stereo = clip.audio.to_soundarray()
        
        n_frames = clip.reader.nframes
        image_id = int(n_frames/2 + randint(0, 30) - 15)  # Take a random central frame in a range of 1 sec (assumes 30 fps)
        frame = clip.get_frame(image_id*1.0/30.0)
        if frame.shape[0] != 360:
            print("DEBUG: Different frame sizes --> {}".format(frame.shape[0]))
            exit()
            #crop = 2 * self.crop_size
        else:
            crop = self.crop_img
        start_x = 0#int(frame.shape[0]/2 - crop/2)
        start_y = int(frame.shape[1]/2 - crop/2)
        image = frame[start_x:start_x+crop,start_y:start_y+crop,:]  # Crop a central area of 280x280
        image = self.totensor(np.array(image))
        image = self.convert(self.resize_image(image))
        
        crop = self.crop_spc
        mono = np.mean(stereo.T, axis=0)
        audio = melspectrogram(y=mono, sr=self.sample_rate, n_fft=2048, win_length=512, hop_length=353, n_mels=crop)
        spectr = power_to_db(audio)
        problem = 0#DEBUG
        if spectr.shape[1] < crop:
            filler = int((crop - spectr.shape[1])/2)
            if np.pad(spectr, ((0,0),(filler,filler)), "reflect").shape[1] != crop:
                spectr = np.pad(spectr, ((0,0),(filler+1,filler)), "reflect")
                problem = 1#DEBUG
            else:
                spectr = np.pad(spectr, ((0,0),(filler,filler)), "reflect")
                problem = 2#DEBUG
        elif spectr.shape[1] > crop:
            remove = int((spectr.shape[1] - crop)/2)
            if spectr[:,remove:-remove].shape[1] != crop:
                spectr = spectr[:,remove:-(remove+1)]
                problem = 3#DEBUG
            else:
                spectr = spectr[:,remove:-remove]
                problem = 4#DEBUG
        spectr_hat = self.convert(self.totensor(spectr))
        if spectr_hat.shape[2] != crop:
            print("DEBUG {}: Different spectrogram sizes --> {}".format(problem, spectr_hat.shape[2]))
            exit()
        
        label_actor = self.labels[0][idx]
        label_sentence = self.labels[1][idx]
        label_emo = self.labels[2][idx]
        label_level = self.labels[3][idx]
        label_age = self.labels[4][idx]
        label_gender = self.labels[5][idx]
        label_race = self.labels[6][idx]
        label_ethnic = self.labels[7][idx]
        
        clip.close()
        
        return image, spectr_hat, label_emo, label_race, label_gender, label_sentence
    