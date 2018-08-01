from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import collections
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')

print "Shape of MNIST Dataset: " + str(mnist.data.shape)

print "Number of samples: " + str(mnist.target.shape)

print "Number of images of each label: " + '\n'  + str(collections.Counter(mnist.target))


train_img, test_img, train_lbl, test_lbl = train_test_split(
 mnist.data, mnist.target, test_size=1/7.0, random_state=0)



plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)


# default solver is incredibly slow thats why we change it
logisticRegr = LogisticRegression(solver = 'lbfgs')


logisticRegr.fit(train_img, train_lbl)


# Returns a NumPy Array
# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))

predictions = logisticRegr.predict(test_img)
i = 0
pos = 0
no = 0
for label in test_lbl:
    if label == predictions[i]:
        pos = pos + 1
    else:
        no = no + 1
    i = i + 1
print "Size of test Dataset: " + str(len(test_img))    
print "Correct Predictions: " + str(pos)
print "Incorrect Predictions: " + str(no)
score = logisticRegr.score(test_img, test_lbl)
print "Accuracy: " + str(score)
