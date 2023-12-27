import numpy as np

from os import makedirs
from os.path import join

# from six.moves import xrange

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
import pickle


def read_variances(data_path):
    with open(join(data_path, "train_image_var.pkl"), "rb") as fp:  # Pickling
        image_var = pickle.load(fp)
    with open(join(data_path, "train_audio_var.pkl"), "rb") as fp:  # Pickling
        audio_var = pickle.load(fp)

    return image_var, audio_var


if __name__ == "__main__":

    # <config>
    message = "Proposed method in the CREMA-D dataset"
    
    timestamp = '20230315-015820'  # None: Train from scratch. Timestamp value: Use trained model
    
    dataset = 'CREMA-D'  # 'EmoVoxCeleb' or 'RML' or 'RML_feat' or 'CREMA-D'. Name of the emotion recognition dataset
    pretrained = False  # True or False. Train from scratch or using pretrained weights
    mode = 'multimodal'  # 'multimodal' or 'image-only' or 'audio-only'
    base_type = 0  # Type of baseline: 0 (no baseline), 1 (no reconstruction nor masking), 2 (no masking), 3 (no reconstruction)

    data_path = "data_cremad"  # 'data' or 'data_rml' or 'data_rml_by_user' or 'data_cremad'. Folder where the config necessary for data loading is stored
    balanced = False  # Loads a class-balanced version of the training data

    num_epoch = 50  # Training epochs. Recommended: 25 (50 if regularization == True)
    batch_size = 128  # Batch size. Recommended: 128
    shuffle = True  # Flag for shuffling the training data on each epoch
    learning_rate = 1e-3  # Learning rate

    num_hiddens = 128  # Number of channels in the convolutional feature maps
    num_residual_hiddens = 32  # Number of channels in the residual feature maps
    num_residual_layers = 2  # Number of residual layers

    embedding_dim = 64  # Size of each code in the VQVAE codebook
    num_embeddings = 512  # Number of codes in the VQVAE codebook
    commitment_cost = 0.25  # Loss term for VQVAE
    decay = 0.99  # Term to update the VQVAE codes when training with EMA (if decay > 0)

    # Parameters for the learnable mask
    mask_scale = 1e-2  # Initial values of the mask
    mask_init = '1s'#'uniform'  # Mask values initialization: all mask_scale, or unif. distr. [-mask_scale,mask_scale]
    threshold_fn = 'binarizer'  # Type of masking: 'binarizer' (0,1), 'ternarizer' (-1,0,1), 'attention' (actual value), 'randomizer' (random 0s and 1s with equal probability aka dropout)
    threshold = None#9e-4#25e-4#5e-3#75e-4#9e-3  # Threshold for which the mask values are binarized/ternarized

    regularization = True  # Add a regularization term to the loss to penalize masks with many '1' values
    modify_modal = False  # Adds irrelevant features (i.e., skin color) to the given task (emotion classification)
    # </config>

    if torch.cuda.is_available():
        device = torch.device("cuda")
        CUDA = True
        num_workers = 4  # (approx.) num_worker = 4 * num_GPU
    else:
        device = torch.device("cpu")
        CUDA = False
        num_workers = 0

    if dataset == 'EmoVoxCeleb':
        from model import Model
        if balanced:
            from datasets_balanced import EmoVoxCelebDatasetBal as EmoDataset
            data_path += "_balanced"
            label_names = ['neutral', 'happiness', 'surprise', 'sadness']
        else:
            from datasets import EmoVoxCelebDataset as EmoDataset
            label_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        criterion = torch.nn.MSELoss()
    elif dataset == 'RML':
        if base_type == 0:
            from model import Model
        else:
            from model import Model_Baseline as Model
        from datasets import RMLDataset as EmoDataset
        label_names = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        criterion = torch.nn.CrossEntropyLoss()
    elif dataset == 'RML_feat':
        from model import Model_Feat as Model
        from datasets import RMLFeatDataset as EmoDataset
        label_names = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        criterion = torch.nn.CrossEntropyLoss()
    elif dataset == 'CREMA-D':
        if base_type == 0:
            from model import Model
        elif base_type == 4:
            from model import Model_Comparison as Model
        else:
            from model import Model_Baseline as Model
        from datasets import CREMADDataset as EmoDataset
        label_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
        criterion = torch.nn.CrossEntropyLoss()
    else:
        print("ERROR - Unknown dataset: {}".format(dataset))
        exit()
    n_classes = len(label_names)  # Number of classes
    
    data_variance1, data_variance2 = read_variances(data_path)

    if dataset == 'RML' and pretrained:
        model = Model('EmoVoxCeleb', mode, num_hiddens, n_classes, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay, mask_init, mask_scale, threshold_fn, threshold).to(device)
        pretrained_model = Model('EmoVoxCeleb', mode, num_hiddens, 8, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay, mask_init, mask_scale, threshold_fn, threshold).to(device)
    else:
        if base_type == 0:
            model = Model(dataset, mode, num_hiddens, n_classes, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay, mask_init, mask_scale, threshold_fn, threshold).to(device)
        elif base_type == 4:
            model = Model(mode, n_classes).to(device)
        else:
            model = Model(dataset, mode, base_type, num_hiddens, n_classes, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay, mask_init, mask_scale, threshold_fn, threshold).to(device)

    log = ""
    
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if CUDA else {}
    
    if not timestamp:

        now = datetime.now()
        timestamp = "{}{}{}-{}{}{}".format(now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), now.strftime("%H"), now.strftime("%M"), now.strftime("%S"))

        # Training
        private1_ratio, private2_ratio, shared1_ratio, shared2_ratio, _, _ = model._classifier.get_private_shared_ratio()
        print_line = "*** INITIAL PRIVATE SHARED SPACE ***\n-Private 1 ratio: {}\n-Private 2 ratio: {}\n-Shared 1 ratio: {}\n-Shared 2 ratio: {}\n".format(private1_ratio, private2_ratio, shared1_ratio, shared2_ratio)
        print(print_line)
        log = log + print_line + "\n"
        print_line = "*** TRAINING ***"
        print(print_line)
        log = log + print_line + "\n"
        
        train_data = EmoDataset(data_path, 'train', pretrained=pretrained, modify_modal=modify_modal)
        val_data = EmoDataset(data_path, 'val', pretrained=pretrained, modify_modal=modify_modal)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
        BIAS_TRAIN = (train_loader.dataset.__len__() - 1) / (batch_size - 1)
        BIAS_VAL = (val_loader.dataset.__len__() - 1) / (batch_size - 1)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if pretrained:
            pretrained_model.load_state_dict(torch.load('results/20220623-051044/model.pt'))
            model._encoder_image = pretrained_model._encoder_image
            model._decoder_image = pretrained_model._decoder_image
            model._encoder_audio = pretrained_model._encoder_audio
            model._decoder_audio = pretrained_model._decoder_audio
            model._pre_vq_conv_image = pretrained_model._pre_vq_conv_image
            model._pre_vq_conv_audio = pretrained_model._pre_vq_conv_audio
            model._vq_vae_image = pretrained_model._vq_vae_image
            model._vq_vae_audio = pretrained_model._vq_vae_audio
        
        train_class_error = []
        train_res_recon_error1 = []
        train_res_recon_error2 = []
        train_res_perplexity1 = []
        train_res_perplexity2 = []
        
        val_class_error = []
        val_res_recon_error1 = []
        val_res_recon_error2 = []
        val_res_perplexity1 = []
        val_res_perplexity2 = []
        
        min_val_loss = 99999999
        min_val_epoch = 0

        NUM_PIXELS = int(28 * 28)
        #torch.autograd.set_detect_anomaly(True)  # Uncomment for DEBUG
        for e in range(num_epoch):
            # for i in xrange(num_training_updates):
            model.train(True)
            i = 0
            for image, audio, labels, _, _, _ in train_loader:
                i += 1

                #if labels.size()[0] == batch_size:
                image = image.to(device)
                audio = audio.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                vq_loss1, vq_loss2, data_recon1, data_recon2, shared_loss, y, perplexity1, perplexity2 = model(image, audio)
                recon_error1 = F.mse_loss(data_recon1, image)# / data_variance1
                recon_error2 = F.mse_loss(data_recon2, audio)# / data_variance2
                class_loss = criterion(y, labels)  # /(batch_size*batch_size)
                # print("Losses:\n-Recon1:{}\n-Recon2:{}\n-VQ1:{}\n-VQ2:{}\n-Classification:{}\n-Shared:{}".format(recon_error1, recon_error2, vq_loss1, vq_loss2, class_loss, shared_loss));exit()
                if not regularization:
                    loss = recon_error1 + recon_error2 + class_loss + vq_loss1 + vq_loss2
                else:
                    loss = recon_error1 + recon_error2 + class_loss + vq_loss1 + vq_loss2 + shared_loss*0.0001#(1/(e+1))
                loss.backward()

                optimizer.step()

                train_class_error.append(class_loss.item())
                train_res_recon_error1.append(recon_error1.item())
                train_res_recon_error2.append(recon_error2.item())
                train_res_perplexity1.append(perplexity1.item())
                train_res_perplexity2.append(perplexity2.item())

                # if i == 5:print("Run for 5 iterations");break#exit()

            print_line = '* Epoch {} - Classification error: {}'.format(e + 1, np.mean(train_class_error[-i:]))
            print(print_line)
            log = log + print_line + "\n"
            print_line = 'Image - Reconstruction error: {}, Perplexity: {}'.format(np.mean(train_res_recon_error1[-i:]), np.mean(train_res_perplexity1[-i:]))
            print(print_line)
            log = log + print_line + "\n"
            print_line = 'Audio - Reconstruction error: {}, Perplexity: {}'.format(np.mean(train_res_recon_error2[-i:]), np.mean(train_res_perplexity2[-i:]))
            print(print_line)
            log = log + print_line + "\n"
            private1_ratio, private2_ratio, shared1_ratio, shared2_ratio, _, _ = model._classifier.get_private_shared_ratio()
            print_line = "Latent space: Private 1 {}, Private 2 {}, Shared 1 {}, Shared 2 {}".format(private1_ratio, private2_ratio, shared1_ratio, shared2_ratio)
            print(print_line)
            log = log + print_line + "\n"

            # Validation
            model.train(False)
            j = 0
            for image, audio, labels, _, _, _ in val_loader:
                j += 1
                #if labels.size()[0] == batch_size:

                image = image.to(device)
                audio = audio.to(device)
                labels = labels.to(device)

                vq_loss1, vq_loss2, data_recon1, data_recon2, shared_loss, y, perplexity1, perplexity2 = model(image, audio)
                recon_error1 = F.mse_loss(data_recon1, image)# / data_variance1
                recon_error2 = F.mse_loss(data_recon2, audio)# / data_variance2
                class_loss = criterion(y, labels)

                val_class_error.append(class_loss.item())
                val_res_recon_error1.append(recon_error1.item())
                val_res_recon_error2.append(recon_error2.item())
                val_res_perplexity1.append(perplexity1.item())
                val_res_perplexity2.append(perplexity2.item())
                
            if np.mean(val_class_error[-j:]) <= min_val_loss:
                # Save model
                makedirs("results/{}".format(timestamp), exist_ok=True)
                torch.save(model.state_dict(), 'results/{}/model.pt'.format(timestamp))
                min_val_loss = np.mean(val_class_error[-j:])
                min_val_epoch = e

            print_line = '* Validation - Classification error: {}'.format(np.mean(val_class_error[-j:]))
            print(print_line)
            log = log + print_line + "\n"
            print_line = 'Image - Reconstruction error: {}, Perplexity: {}'.format(np.mean(val_res_recon_error1[-j:]), np.mean(val_res_perplexity1[-j:]))
            print(print_line)
            log = log + print_line + "\n"
            print_line = 'Audio - Reconstruction error: {}, Perplexity: {}\n'.format(np.mean(val_res_recon_error2[-j:]), np.mean(val_res_perplexity2[-j:]))
            print(print_line)
            log = log + print_line + "\n"

        # Save log
        with open('results/{}/log.txt'.format(timestamp), 'w') as f:
            f.write("{}\n\n".format(message))
            f.write("Model saved in epoch {}:\n".format(min_val_epoch+1))
            f.write("CONFIG:\n")
            f.write("- Dataset: {}\n".format(dataset))
            f.write("- Pretrained: {}\n".format(pretrained))
            f.write("- Mode: {}\n".format(mode))
            f.write("- Baseline: {}\n".format(base_type))
            f.write("- Data path: {}\n".format(data_path))
            f.write("- Num epoch: {}\n".format(num_epoch))
            f.write("- Batch size: {}\n".format(batch_size))
            f.write("- Learning rate: {}\n".format(learning_rate))
            f.write("- Num hiddens: {}\n".format(num_hiddens))
            f.write("- Num residual hiddens: {}\n".format(num_residual_hiddens))
            f.write("- Num residual layers: {}\n".format(num_residual_layers))
            f.write("- Embedding dimension: {}\n".format(embedding_dim))
            f.write("- Num embeddings: {}\n".format(num_embeddings))
            f.write("- Commitment cost: {}\n".format(commitment_cost))
            f.write("- Decay: {}\n".format(decay))
            f.write("- Mask init: {}\n".format(mask_init))
            f.write("- Mask scale: {}\n".format(mask_scale))
            f.write("- Threshold function: {}\n".format(threshold_fn))
            f.write("- Threshold: {}\n".format(threshold))
            f.write("- Regularization: {}\n".format(regularization))
            f.write(log)

        # Save training stats for visualization in Notebook

        with open("results/{}/train_res_recon_error1.pkl".format(timestamp), "wb") as fp:  # Pickling
            pickle.dump(train_res_recon_error1, fp)
        with open("results/{}/train_res_recon_error2.pkl".format(timestamp), "wb") as fp:  # Pickling
            pickle.dump(train_res_recon_error2, fp)

        with open("results/{}/train_res_perplexity1.pkl".format(timestamp), "wb") as fp:  # Pickling
            pickle.dump(train_res_perplexity1, fp)
        with open("results/{}/train_res_perplexity2.pkl".format(timestamp), "wb") as fp:  # Pickling
            pickle.dump(train_res_perplexity2, fp)

        with open("results/{}/train_class_error.pkl".format(timestamp), "wb") as fp:  # Pickling
            pickle.dump(train_class_error, fp)

        with open("results/{}/val_res_recon_error1.pkl".format(timestamp), "wb") as fp:  # Pickling
            pickle.dump(val_res_recon_error1, fp)
        with open("results/{}/val_res_recon_error2.pkl".format(timestamp), "wb") as fp:  # Pickling
            pickle.dump(val_res_recon_error2, fp)

        with open("results/{}/val_res_perplexity1.pkl".format(timestamp), "wb") as fp:  # Pickling
            pickle.dump(val_res_perplexity1, fp)
        with open("results/{}/val_res_perplexity2.pkl".format(timestamp), "wb") as fp:  # Pickling
            pickle.dump(val_res_perplexity2, fp)

        with open("results/{}/val_class_error.pkl".format(timestamp), "wb") as fp:  # Pickling
            pickle.dump(val_class_error, fp)
            
        if mode == 'multimodal' and base_type != 4:
            with open("results/{}/weight1.pkl".format(timestamp), "wb") as fp:  # Pickling
                pickle.dump(model._vq_vae_image._embedding.weight.data.cpu(), fp)
            with open("results/{}/weight2.pkl".format(timestamp), "wb") as fp:  # Pickling
                pickle.dump(model._vq_vae_audio._embedding.weight.data.cpu(), fp)

    # Load the best configuration
    model.load_state_dict(torch.load('results/{}/model.pt'.format(timestamp)))

    # Test
    print_line = "*** EVALUATION ***"
    print(print_line)
    log = print_line + "\n"

    test_data = EmoDataset(data_path, 'test', pretrained=pretrained, modify_modal=modify_modal)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
    BIAS_TEST = (test_loader.dataset.__len__() - 1) / (batch_size - 1)

    model.eval()

    correct = np.zeros(n_classes)  # 1
    total = np.zeros(n_classes)  # 1
    # 2error = np.zeros(n_classes)
    # 2total = 0
    with torch.no_grad():

        for image, audio, labels, _, _, _ in test_loader:

            #if labels.size()[0] == batch_size:

            data_orig1 = image.to(device)
            data_orig2 = audio.to(device)
            # labels = labels.to(device)

            _, _, data_recon1, data_recon2, _, y, _, _ = model(data_orig1, data_orig2)
            # collect the correct predictions for each class
            _, predictions = torch.max(y.cpu(), 1)  # 1
            if dataset == 'EmoVoxCeleb':
                _, labels = torch.max(labels, 1)  # 1
            for prediction, label in zip(predictions, labels):  # 1
                if label == prediction:  # 1
                    correct[label] += 1  # 1
                total[label] += 1
            # 2for prediction, label in zip(y, labels):
            # 2    for c in range(n_classes):
            # 2       error[c] += np.abs(label[c] - prediction[c].cpu().numpy())
            # 2total += len(labels)

    # print accuracy for each class
    for c in range(n_classes):
        accuracy_class = 100 * float(correct[c]) / total[c]  # 1
        print_line = "Accuracy for class {}: {}".format(label_names[c], accuracy_class)  # 1
        print(print_line)  # 1
        log = log + print_line + "\n"  # 1
    accuracy_total = 100 * np.sum(correct) / np.sum(total)  # 1
    print_line = "System accuracy: {}".format(accuracy_total)  # 1
        # 2mean_error = float(error[c]) / total
        # 2print_line = "Mean error for class {}: {}".format(label_names[c], mean_error)
        # 2print(print_line)
        # 2log = log + print_line + "\n"
    # 2error_total = np.sum(error) / (total * n_classes)
    # 2print_line = "Total mean error: {}".format(error_total)  #
    print(print_line)
    log = log + print_line + "\n"
    private1_ratio, private2_ratio, shared1_ratio, shared2_ratio, compl, _ = model._classifier.get_private_shared_ratio()
    print_line = "\n*** LEARNED PRIVATE SHARED SPACE ***\n-Private 1 ratio: {}\n-Private 2 ratio: {}\n-Shared 1 ratio: {}\n-Shared 2 ratio: {}\n-Complementarity: {}\n".format(private1_ratio, private2_ratio, shared1_ratio, shared2_ratio, compl)
    print(print_line)
    log = log + print_line + "\n"

    with open('results/{}/log.txt'.format(timestamp), 'a') as f:
        f.write(log)

    # Save results for visualization in Notebook

    with open("results/{}/valid_reconstructions1.pkl".format(timestamp), "wb") as fp:
        pickle.dump(data_recon1.cpu().data, fp)  # Pickling
    with open("results/{}/valid_reconstructions2.pkl".format(timestamp), "wb") as fp:
        pickle.dump(data_recon2.cpu().data, fp)  # Pickling

    with open("results/{}/valid_originals1.pkl".format(timestamp), "wb") as fp:
        pickle.dump(data_orig1.cpu(), fp)  # Pickling
    with open("results/{}/valid_originals2.pkl".format(timestamp), "wb") as fp:
        pickle.dump(data_orig2.cpu(), fp)  # Pickling
