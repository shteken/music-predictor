import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree as tre
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools

names=['Song Name',
       'non zero frequencies',
       'most repeated frequency',
       'average distance frequencies',
       'std distance frequencies',
       'number of non zero times',
       'time with the most frequencies',
       'average distance times',
       'std distance times']
parameters = pd.read_csv('music.csv', header=None, names=names)
# print(parameters)

names2 = ['Song Name','preference']
preference = pd.read_csv('preferences.csv', header=None, names=names2)
# print(preference)

df = parameters.merge(preference, on='Song Name')
print(df)

#####Machine Learning Engine!!!#####
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# print(X_train, X_test, y_train, y_test)

print(list(y_test))
print('______')
#####KNN#####
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(confusion_matrix(y_test, knn.predict(X_test), labels=[1, 0]))
print(accuracy_score(y_test, knn.predict(X_test)))


#####SVM#####
svm = SVC(random_state=0, kernel='rbf')
svm.fit(X_train, y_train)
print(svm.predict(X_test))
print(confusion_matrix(y_test, svm.predict(X_test), labels=[1, 0]))
print(accuracy_score(y_test, svm.predict(X_test)))


#####Decision Tree#####
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print(tree.predict(X_test))
print(confusion_matrix(y_test, tree.predict(X_test), labels=[1, 0]))
print(accuracy_score(y_test, tree.predict(X_test)))



one_attribute_df = df.iloc[:, 1]
dont_like_df = df[df.iloc[:, -1] == 0]
one_attribute_dont_like_df = dont_like_df.iloc[:, 1]
like_df = df[df.iloc[:, -1] == 1]
one_attribute_like_df = like_df.iloc[:, 1]
two_attribute_df = df.iloc[:, 2]
two_attribute_dont_like_df = dont_like_df.iloc[:, 2]
two_attribute_like_df = like_df.iloc[:, 2]

indexes_dont_like = list(one_attribute_dont_like_df.index)
indexes_like = list(one_attribute_like_df.index)

def onpick(event):
    print(event.artist.get_label())
    for ind in event.ind:
        if event.artist.get_label() == 'like':
            print(df.iloc[indexes_like[ind], 0])
        else:
            print(df.iloc[indexes_dont_like[ind], 0])

var = [
       'non zero frequencies',
       # 'most repeated frequency', #
       'average distance frequencies',
       'std distance frequencies',
       # 'number of non zero times', #
       # 'time with the most frequencies', #
       'average distance times',
       'std distance times']

permuted_parameters = list(itertools.permutations(var, 2))
print(permuted_parameters)

# sns.pairplot(df, hue='preference', vars=var)
# plt.show()
fig = plt.figure()


def enter(sen):
    words = sen.split()
    return (words[0] + ' ' + words[1] + '\n' + words[2])


def make_hist(i_am_hist):
    ax1 = fig.add_subplot(5,5,i_am_hist)
    one_attribute_dont_like_df = dont_like_df[var[diagonal.index(i_am_hist)]]
    one_attribute_like_df = like_df[var[diagonal.index(i_am_hist)]]
    # maximum = max(max(one_attribute_dont_like_df), max(one_attribute_like_df))
    # minimum = min(min(one_attribute_dont_like_df), min(one_attribute_like_df))
    # bins = np.linspace(minimum, maximum, 11)
    stacked = [one_attribute_like_df, one_attribute_dont_like_df]
    if i_am_hist == 1:
        ax1.set_title(var[0], size=10)
        plt.ylabel(enter(var[0]), size=9)
    ax1.hist(stacked, bins=10, stacked=True, color=['royalblue', 'magenta'])
    # ax1.hist(one_attribute_like_df, bins=bins, color='b', alpha=0.7, stacked=True)
    # ax1.hist(one_attribute_dont_like_df, bins=bins, color='r', alpha=0.5, stacked=True)


def make_scatter(i):
    ax1 = fig.add_subplot(5,5,i)
    one_attribute_dont_like_df = dont_like_df[permutation[0]]
    one_attribute_like_df = like_df[permutation[0]]
    two_attribute_dont_like_df = dont_like_df[permutation[1]]
    two_attribute_like_df = like_df[permutation[1]]
    if i < 6 and i != 1:
        ax1.set_title(permutation[1], size=10)
    if (i-1)%5 == 0:
        plt.ylabel(enter(permutation[0]), size=9)
    ax1.plot(one_attribute_dont_like_df, two_attribute_dont_like_df, 'v', color='magenta',picker=3, ms=7, label='dont_like', alpha=0.5)
    ax1.plot(one_attribute_like_df, two_attribute_like_df, '^', color='royalblue', picker=3, ms=7, label='like', alpha=0.5)


##### make scatters of all parameters! #####
i = 1
diagonal = []
for permutation in permuted_parameters:
    if ((i-1)%6 == 0) or i==1:
        diagonal.append(i)
        i += 1
    make_scatter(i)
    i += 1
diagonal.append(i)
print(diagonal)


##### make histograms! #####
for i_am_hist in diagonal:
    make_hist(i_am_hist)

fig.canvas.mpl_connect('pick_event', onpick)
plt.show()
