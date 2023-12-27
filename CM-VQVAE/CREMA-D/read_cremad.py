import numpy as np
from os import makedirs, listdir
from os.path import join, exists
import pickle
from csv import reader as csv_reader
import random

from torch.utils.data import DataLoader

from datasets import CREMADDataset


def get_age_class(age):
    
    age_bin = ''
    if age >= 60:
        age_bin = '60s'
    elif age >= 50:
        age_bin = '50s'
    elif age >= 40:
        age_bin = '40s'
    elif age >= 30:
        age_bin = '30s'
    elif age >= 20:
        age_bin = '20s'
    else:
        print("Age out of range: {}".format(age))
        exit()
        
    return age_bin


def read_samples(root_path, video_path):
    
    with open(join(root_path,'VideoDemographics.csv')) as info_file:
        info_reader = csv_reader(info_file, delimiter=',')
        next(info_reader, None)  # skip the headers
        actor_info = {row[0][-2:]:[int(row[1]), row[2], row[3], row[4]] for row in info_reader}
    
    with open(join(root_path,'SentenceFilenames.csv')) as samples_file:
        samples_reader = csv_reader(samples_file, delimiter=',')
        next(samples_reader, None)  # skip the headers
        samples = [join(root_path, video_path, row[1])+'.flv' for row in samples_reader]
    
    #random.seed(0)
    random.shuffle(samples)
    
    return actor_info, samples
    

def create_config(samples, actor_info, data_path, label_names, split):
    
    faulty = ['1064_TIE_SAD_XX.flv','1064_IEO_DIS_MD.flv','1076_MTI_NEU_XX.flv','1076_MTI_SAD_XX.flv']
    
    labels = [[],[],[],[],[],[],[],[]]
    label_actor_ratio = np.zeros(len(label_names['actor']))
    label_sentence_ratio = np.zeros(len(label_names['sentence']))
    label_emo_ratio = np.zeros(len(label_names['emo']))
    label_level_ratio = np.zeros(len(label_names['level']))
    label_age_ratio = np.zeros(len(label_names['age']))
    label_gender_ratio = np.zeros(len(label_names['gender']))
    label_race_ratio = np.zeros(len(label_names['race']))
    label_ethnic_ratio = np.zeros(len(label_names['ethnic']))
    
    for sample in samples:
        video = sample.rsplit('/', 1)[-1]
        if video in faulty:
            continue
        actor = video[2:4]
        sentence = video[5:8]
        emo = video[9:12]
        level = video[13:15]
        age = get_age_class(actor_info[actor][0])
        gender = actor_info[actor][1]
        race = actor_info[actor][2]
        ethnic = actor_info[actor][3]

        label_id = label_names['actor'].index(actor)
        labels[0].append(label_id)
        label_actor_ratio[label_id]+=1
        label_id = label_names['sentence'].index(sentence)
        labels[1].append(label_id)
        label_sentence_ratio[label_id]+=1
        label_id = label_names['emo'].index(emo)
        labels[2].append(label_id)
        label_emo_ratio[label_id]+=1
        label_id = label_names['level'].index(level)
        labels[3].append(label_id)
        label_level_ratio[label_id]+=1
        label_id = label_names['age'].index(age)
        labels[4].append(label_id)
        label_age_ratio[label_id]+=1
        label_id = label_names['gender'].index(gender)
        labels[5].append(label_id)
        label_gender_ratio[label_id]+=1
        label_id = label_names['race'].index(race)
        labels[6].append(label_id)
        label_race_ratio[label_id]+=1
        label_id = label_names['ethnic'].index(ethnic)
        labels[7].append(label_id)
        label_ethnic_ratio[label_id]+=1
        
    makedirs(data_path, exist_ok=True)
    with open(join(data_path, split + "_videos.pkl"), "wb") as fp:  # Pickling
        pickle.dump(samples, fp)
    with open(join(data_path, split + "_labels.pkl"), "wb") as fp:  # Pickling
        pickle.dump(labels, fp)

    label_ratios = {'actor':label_actor_ratio, 'sentence':label_sentence_ratio, 'emo':label_emo_ratio, 'level':label_level_ratio, 'age':label_age_ratio, 'gender':label_gender_ratio, 'race':label_race_ratio, 'ethnic':label_ethnic_ratio}
        
    return label_ratios


def print_class_ratio(label_ratios, label_names, split):

    with open(join(data_path, split + "_stats.txt"), 'w') as f:
        f.write("\n".format(split.upper()))
        print(split.upper())
        for key in label_ratios.keys():
            f.write("* {}\n".format(key))
            print("* {}".format(key))
            label_ratio = label_ratios[key]
            label_total = np.sum(label_ratio)
            for l, label in enumerate(label_names[key]):
                f.write("- Class {}: {} samples, {:.2f} %\n".format(label, int(label_ratio[l]), 100*label_ratio[l]/label_total))
                print("- Class {}: {} samples, {:.2f} %".format(label, int(label_ratio[l]), 100*label_ratio[l]/label_total))
            f.write("Total {} {} samples: {}\n".format(key, split, int(label_total)))
            print("Total {} {} samples: {}".format(key, split, int(label_total)))

    
if __name__ == "__main__":

    # <config>
    root_path = join('..','..','datasets','CREMA-D','CREMA-D')
    video_path = 'VideoFlash'
    split_idx = {'train':(0, 5954), 'val':(5954, 6698), 'test':(6698, 7442)}
    splits = ['train','val','test']
    data_path = "data_cremad"
    # </config>
    
    label_actor_names = [str(i).zfill(2) for i in range(1,92)]
    label_sentence_names = ['IEO', 'TIE', 'IOM', 'IWW', 'TAI', 'MTI', 'IWL', 'ITH', 'DFA', 'ITS', 'TSI', 'WSI']
    label_emo_names = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']  # anger, disgust, fear, happy, neutral, sad
    label_level_names = ['LO', 'MD', 'HI', 'XX']  # low, medium, high, unspecified
    label_age_names = ['20s', '30s', '40s', '50s', '60s']
    label_gender_names = ['Male', 'Female']
    label_race_names = ['African American', 'Asian', 'Caucasian', 'Unknown']
    label_ethnic_names = ['Not Hispanic', 'Hispanic']

    label_names = {'actor': label_actor_names, 'sentence': label_sentence_names, 'emo': label_emo_names, 'level': label_level_names, 'age': label_age_names, 'gender': label_gender_names, 'race': label_race_names, 'ethnic': label_ethnic_names}

    makedirs(data_path, exist_ok=True)
    if not listdir(data_path):
        print("Creating config data...")
        actor_info, samples = read_samples(root_path, video_path)
        for split in splits:
            init, end = split_idx[split]
            label_ratios = create_config(samples[init:end], actor_info, data_path, label_names, split)
            print_class_ratio(label_ratios, label_names, split)
    print("...config data created")

    num_samples = split_idx['train'][1]  # Total number of train samples
    
    train_data = CREMADDataset(data_path, split='train', None, None)
    train_loader = DataLoader(train_data, batch_size=num_samples, shuffle=False)

    for data in train_loader:
        image, audio, _ = data

        max_image = 255.0
        max_audio = np.max(audio.numpy())

        v_image = np.var(image.numpy() / max_image)
        v_audio = np.var(audio.numpy() / max_audio)

    makedirs(data_path, exist_ok=True)
    with open(join(data_path, "train_image_var.pkl"), "wb") as fp:  # Pickling
        pickle.dump(v_image, fp)
    with open(join(data_path, "train_audio_var.pkl"), "wb") as fp:  # Pickling
        pickle.dump(v_audio, fp)

    print("Image: max={}, variance={}\nAudio: max={}, variance={}\n".format(max_image, v_image, max_audio, v_audio))
    # Image: max=255.0, variance=6.30948477464699e-07
    # Audio: max=11.957438468933105, variance=2.3635575771331787
