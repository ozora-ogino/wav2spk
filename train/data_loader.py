import os
import pickle
import numpy as np
import librosa


def get_data(protocol, fs=16000, return_fs=False, length=30916):
    data_list = []
    for i in range(len(protocol)):
        data, _ = librosa.load(protocol[i], sr=fs)

        if len(data) < length:
            diff = length - len(data)
            data = np.append(data, np.zeros(diff))
        else:
            data = data[:length]

        data_list.append(data)
    print(data.shape)
    print(np.array(data_list).shape)

    if return_fs:
        return np.array(data_list), fs
    else:
        return np.array(data_list)


def get_2ch_data(ch1, ch2, fs=False, length=30916):
    data_list = []
    for i in range(len(ch1)):
        ch1_data, _ = librosa.load(ch1[i], sr=16000)
        ch2_data, _ = librosa.load(ch2[i], sr=16000)

        if len(ch1_data) < length:
            diff = length - len(ch1_data)
            ch1_data = np.append(ch1_data, np.zeros(diff))

        if len(ch2_data) < length:
            diff = length - len(ch2_data)
            ch2_data = np.append(ch2_data, np.zeros(diff))

        data_combined = np.array([ch1_data[:length], ch2_data[:length]])
        data_list.append(data_combined)

    print(f"data shape: {np.array(data_list).shape}")

    if fs:
        return np.array(data_list), fs
    else:
        return np.array(data_list)


def get_train_from_binary(lenght=30916):
    path_ge = ""
    path_sp = ""
    with open(path_ge, "rb") as f:
        ge = pickle.load(f)
    with open(path_sp, "rb") as f:
        sp = pickle.load(f)

    data = np.concatenate([ge, sp], axis=0)
    label = np.zeros(len(data))
    label[len(ge) :] = 1
    return data, label.astype(int)


def get_test_data(length=30916):
    def get_protocol(protocol):
        protocol_dir = "../../2ch/features/irt/protocol/"
        protocol = protocol_dir + protocol
        with open(protocol, "r") as f:
            data = f.read()
        data = data.split()
        return data

    protocol_ge_te_1 = "ge_te_ch1.txt"
    protocol_ge_te_2 = "ge_te_ch2.txt"
    protocol_sp_te_1 = "sp_te_ch1.txt"
    protocol_sp_te_2 = "sp_te_ch2.txt"

    # case: 1ch
    protocol_ge = get_protocol("ge_te_ch1.txt")
    protocol_sp = get_protocol("sp_te_ch1.txt")

    ge = get_data(protocol_ge, length=length)
    sp = get_data(protocol_sp, length=length)
    data = np.concatenate([ge, sp], axis=0)
    data = data[:, :, np.newaxis]
    label = np.zeros(len(data))
    label[len(ge) :] = 1

    return data, label.astype(int)


def get_dataset(path_ge, path_sp):
    with open(path_ge, "rb") as f:
        ge = pickle.load(f)

    with open(path_sp, "rb") as f:
        sp = pickle.load(f)

    data = np.concatenate([ge, sp], axis=0)
    data = data[:, :, np.newaxis]
    labels = np.zeros(len(data))
    labels[len(ge) :] = 1
    return data, labels.astype(int)
