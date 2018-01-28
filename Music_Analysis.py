import librosa
import librosa.display as dsp
import matplotlib.pyplot as plt
import numpy as np
import csv
import ntpath

songs = [
    # list here the songs you want to analyze
]

songs_parameters = {}

def bin_matrix(matrix, bin_size=10):
    bin_vector = []
    avg_vector = []
    for x in matrix:
        if len(bin_vector)<bin_size:
            bin_vector.append(x)
        else:
            avg_vector.append(np.average(bin_vector, axis=0))
            bin_vector = []
    return np.array(avg_vector)

def harmonics(processed_matrix, treshold=20):
    number_of_harmonics = []
    counter = 0
    for x in processed_matrix:
        for y in x:
            if y>=treshold:
                counter += 1
        number_of_harmonics.append(counter)
        counter = 0
    return number_of_harmonics

def counter2(vector):
    counter = 0
    for value in vector:
        if value != 0:
            counter += 1
    return counter

def counter(vector):
    return len(list(filter(lambda value: value != 0, vector)))

def weighted_avg_freq(vector):
    average = 0
    place = 0
    sum_place = 0
    for value in vector:
        place += 1
        sum_place += value
        average += value*place
    average /= sum_place
    return average

def max_freq(vector):
    return np.argmax(vector)

def distance(vector, treshold=100):
    distances = []
    precede_value = 0
    latter_value = 0
    current_distance = 1
    for value in vector:
        if value >= treshold:
            precede_value = latter_value
            latter_value = value
            distances.append(current_distance)
            current_distance = 1
        else:
            current_distance += 1
    return distances

def append_dic(value):
    if ntpath.basename(filename) in songs_parameters:
        songs_parameters[ntpath.basename(filename)].append(value)
    else:
        songs_parameters[ntpath.basename(filename)] = [value]

# read the csv file
with open('music.csv', 'r') as fp:
    a = csv.reader(fp)
    analyzed_songs = [line[0] for line in a]

for filename in songs:
    if ntpath.basename(filename) in analyzed_songs:
        continue
    y, sr = librosa.load(filename)
    print(filename)
    # print(sr)
    print('stft')
    D = librosa.stft(y)
    D_abs = np.absolute(D)
    # librosa.display.specshow(bin_matrix(matrix=D.transpose(),bin_size=8).transpose(), y_axis='log', x_axis='time')
    # librosa.display.specshow(D_abs, y_axis='log', x_axis='time')
    # plt.title('Power spectrogram')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    print('harmonics')
    hs = harmonics(processed_matrix=D_abs)
    hsT = harmonics(processed_matrix=D_abs.transpose())
    print('finished harmonics')

    # Over frequencies (ignore of time)
    append_dic(counter(hs))
    append_dic(max_freq(hs))
    append_dic(np.mean(distance(hs)))
    append_dic(np.std(distance(hs)))
    # print('number of repeats in each frequency:{}'.format(hs))
    print('number of non zero frequencies:{}'.format(counter(hs)))
    # print('weighted average of all frequencies:{}'.format(weighted_avg_freq(hs)))
    print('most repeated frequency:{}'.format(max_freq(hs)))
    # print('distance between 2 adjacent frequencies:{}'.format(distance(hs)))
    print('average distance between 2 adjacent frequencies:{}'.format(np.mean(distance(hs))))
    print('std distance between 2 adjacent frequencies:{}'.format(np.std(distance(hs))))

    # Over time(ingnore values of frequencies)
    append_dic(counter(hsT))
    append_dic(max_freq(hsT))
    append_dic(np.mean(distance(hsT, treshold=10)))
    append_dic(np.std(distance(hsT, treshold=10)))
    # print('number of frequencies in each time:{}'.format(hsT))
    print('number of non zero times:{}'.format(counter(hsT)))
    # print('weighted average of all frequencies:{}'.format(weighted_avg_freq(hsT)))
    print('the time with the most frequencies:{}'.format(max_freq(hsT)))
    # print('distance between 2 adjacent frequencies:{}'.format(distance(hsT, treshold=10)))
    print('average distance between 2 adjacent times:{}'.format(np.mean(distance(hsT, treshold=10))))
    print('std distance between 2 adjacent times:{}'.format(np.std(distance(hsT, treshold=10))))
    # plt.plot(hs)
    # plt.bar(left=range(len(hs)), height=hs)
    # plt.show()
    # plt.bar(left=range(len(hsT)), height=hsT)
    # plt.show()

songs_parameters_list = [ [p[0]] + p[1] for p in songs_parameters.items() ]

with open('music.csv', 'a', newline='') as fp:
    a = csv.writer(fp)
    a.writerows(songs_parameters_list)
print(songs_parameters)
