import json
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ReadJSON():
    with open("/home/engin/Documents/MSc/CENG574/featuresTiny.json", "r") as fileHandle:
        content = fileHandle.read()
    jsonObj = json.loads(content)
    jsonImages = jsonObj["images"]


    counter = 0
    samples = [0,1,2,3,4,40,41,42,43,44,80,81,82,83,84,120,121,122,123,124,160,161,162,163,164,
               200, 201, 202, 203, 204, 240,241,242,243,244,280,281,282,283,284]
    with open("/home/engin/Documents/MSc/CENG574/featuresTiny.csv", "w") as writeHandle:
        for jsonImage in jsonImages:
            if counter in samples:
                lineStr = ""
                for feature in jsonImage["features"]:
                    lineStr = lineStr + str(feature) + ","
                lineStr = lineStr[:-1] + "\n"
                writeHandle.write(lineStr)
            counter = counter + 1

def ReadJSON2():
    with open("/home/engin/Documents/MSc/CENG574/featuresTiny.json", "r") as fileHandle:
        content = fileHandle.read()
    jsonObj = json.loads(content)
    jsonImages = jsonObj["images"]

    features = []
    #samples = [0,1,2,3,4,40,41,42,43,44,80,81,82,83,84,120,121,122,123,124]
    samples = []
    #(classNumber - 1) * 40 + 1
    for a in range(0,761, 40):
        counter = 0
        while True:
            samples.append(a + counter)
            counter = counter + 1
            if counter == 5:
                break

    counter = 0
    for jsonImage in jsonImages:
        if counter in samples:
            features.append(jsonImage["features"])
            print (counter)
        counter = counter + 1
    return features

def Plot2D(data):
    plt.figure(1)
    plt.clf()
    # plt.imshow(Z, interpolation='nearest',
    #            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #            cmap=plt.cm.Paired,
    #            aspect='auto', origin='lower')


    counter = 0
    markers = [('o', 'blue'), ('v', 'orange'), ('^', 'green'), ('<', 'red'), ('>', 'brown'),
               ('1', 'blue'), ('2', 'orange'), ('3', 'green'), ('4', 'red'), ('8', 'brown')]
    #classnumber * 5
    for i in range(0, 50, 5):
        block = data[i:i+5]
        (marker1, marker2) = markers[counter]
        plt.scatter(block[:, 0], block[:, 1], marker=marker1, s=25, linewidths=3, color=marker2, zorder=10)
        counter = counter + 1

def Plot3D(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    markers = [('o', 'blue'), ('v', 'orange'), ('^', 'green'), ('<', 'red'), ('>', 'brown'),
               ('1', 'blue'), ('2', 'orange'), ('3', 'green'), ('4', 'red'), ('8', 'brown')]

    #classnumber * 5
    counter = 0
    for i in range(0, 50, 5):
        block = data[i:i+5]
        (marker1, marker2) = markers[counter]
        ax.scatter(block[:, 0], block[:, 1], block[:, 2], c=marker2, marker=marker1)
        counter = counter + 1

    return None

if __name__ == "__main__":
    features = ReadJSON2()
    reduced_data = PCA(n_components=3).fit_transform(features)
    # x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    # y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

    #Plot2D(reduced_data)
    Plot3D(reduced_data)

    a = 1


