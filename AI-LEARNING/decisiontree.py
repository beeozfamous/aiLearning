import idx2numpy
import seaborn as sns

X_train_3D = idx2numpy.convert_from_file('image/train-images-idx3-ubyte')
X_train = X_train_3D.flatten().reshape(60000,784)

y_train = idx2numpy.convert_from_file('image/train-labels-idx1-ubyte')

X_test_3D = idx2numpy.convert_from_file('image/t10k-images-idx3-ubyte')
X_test =  X_test_3D.flatten().reshape(10000,784)

y_test = idx2numpy.convert_from_file('image/t10k-labels-idx1-ubyte')

import matplotlib
import matplotlib.pyplot as plt



def display(image, label):
    """image is a 1*784 numpy array"""

    image = image.reshape(28, 28)
    sns.heatmap(image, linewidth=0, xticklabels=False, yticklabels=False)
    # plt.imshow(image, cmap = plt.cm.gray_r, interpolation="nearest")
    plt.title("Image Representation for %d" % (label))
    plt.show()

for i in range(10):
    display(X_train[i],y_train[i])

from sklearn.utils import shuffle
X_shuffle,y_shuffle = shuffle(X_train,y_train)
X_train = X_shuffle[0:50000]
y_train = y_shuffle[0:50000]

from sklearn import tree
from sklearn.model_selection import cross_val_predict


dt_clf = tree.DecisionTreeClassifier()

y_train_pred = cross_val_predict(dt_clf, X_train, y_train, cv=3)
dt_clf.fit(X_train, y_train)