import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from shutil import copyfile

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
    with open("/home/engin/Documents/CENG574Project/featuresTiny.json", "r") as fileHandle:
        content = fileHandle.read()
    jsonObj = json.loads(content)
    jsonImages = jsonObj["images"]

    features = []
    #samples = [0,1,2,3,4,40,41,42,43,44,80,81,82,83,84,120,121,122,123,124]
    samples = []
    #(classNumber - 1) * 40 + 1
    for a in range(0,561, 40):
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

def SplitData():
    with open("/home/engin/Documents/CENG574Project/featuresTiny.json", "r") as fileHandle:
        content = fileHandle.read()
    jsonObj = json.loads(content)
    jsonImages = jsonObj["images"]

    #Split the images as their classes
    imagesOfClasses =[]
    pathsOfClasses = []
    namesOfClasses = []
    idsOfClasses = []
    for i in range(0,2000, 40):
        imagesOfSingleClass = []
        pathsOfSingleClass = []
        for jsonImage in jsonImages[i:i + 40]:
            imagesOfSingleClass.append(jsonImage["features"])
            pathsOfSingleClass.append(jsonImage["path"])
        namesOfClasses.append(jsonImages[i:i+40][0]["class_name"])
        idsOfClasses.append(jsonImages[i:i + 40][0]["class_id"])
        imagesOfClasses.append(imagesOfSingleClass)
        pathsOfClasses.append(pathsOfSingleClass)

    #Split images as train and test data
    train = []
    test = []

    for imagesOfClass in imagesOfClasses:
        train.append(imagesOfClass[0:28])
        test.append(imagesOfClass[28:40])

    train_paths = []
    test_paths = []
    for pathsOfClass in pathsOfClasses:
        train_paths.append(pathsOfClass[0:28])
        test_paths.append(pathsOfClass[28:40])


    return (train, test, namesOfClasses, idsOfClasses, train_paths, test_paths)


def ExtractLabelsForClasses(model):
    classLabels = []
    for i in range(0,1400, 28):
        label = np.argmax(np.bincount(model.labels_[i:i+28]))
        classLabels.append(label)
    return classLabels

def GetWrongPaths(paths, booleanArray):
    wrongPaths = []
    for (path,boolean) in zip(paths, booleanArray):
        if not boolean:
            wrongPaths.append(path)
    return wrongPaths

def CpWrong(wrongPaths, classLabel):
    os.mkdir(os.path.dirname("./" + str(classLabel) + "/"))
    for wrongPath in wrongPaths:
        dest = "./" + str(classLabel) + "/" + os.path.basename(wrongPath)
        copyfile(wrongPath, dest)

def TestModel(model, testData, pathsOfTest):
    totalCorrects = 0
    classCounter = 0
    classLabels = ExtractLabelsForClasses(model)
    for i in range(0,600, 12):
        imageForClass = testData[i:i+12]
        subsetPaths = pathsOfTest[classCounter]
        result = model.predict(imageForClass)
        #wrongPaths = GetWrongPaths(subsetPaths, result == classLabels[classCounter])
        #CpWrong(wrongPaths, classCounter)
        result = result[result == classLabels[classCounter]]
        totalCorrects = totalCorrects + result.shape[0]
        classCounter = classCounter + 1
    return totalCorrects / 600.0



def Plot2D(data):
    plt.figure(1)
    plt.clf()
    # plt.imshow(Z, interpolation='nearest',
    #            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #            cmap=plt.cm.Paired,
    #            aspect='auto', origin='lower')


    counter = 0
    markers = [('p', 'blue'), ('P', 'orange'), ('h', 'green'), ('H', 'red'), ('*', 'brown'),
                ('o', 'blue'), ('v', 'orange'), ('^', 'green'), ('<', 'red'), ('>', 'brown'),
               ('1', 'blue'), ('2', 'orange'), ('3', 'green'), ('4', 'red'), ('8', 'brown')
               ]
    #classnumber * 5
    for i in range(0, 75, 5):
        block = data[i:i+5]
        (marker1, marker2) = markers[counter]
        plt.scatter(block[:, 0], block[:, 1], marker=marker1, s=25, linewidths=3, color=marker2, zorder=10)
        counter = counter + 1

def Plot3D(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    markers = [('o', 'blue'), ('v', 'orange'), ('^', 'green'), ('<', 'red'), ('>', 'brown'),
               ('1', 'blue'), ('2', 'orange'), ('3', 'green'), ('4', 'red'), ('8', 'brown'),
               ('p', 'blue'), ('P', 'orange'), ('h', 'green'), ('H', 'red'), ('*', 'brown')]

    #classnumber * 5
    counter = 0
    for i in range(0, 75, 5):
        block = data[i:i+5]
        (marker1, marker2) = markers[counter]
        ax.scatter(block[:, 0], block[:, 1], block[:, 2], c=marker2, marker=marker1)
        counter = counter + 1

    return None

def FindMisclassified(model):
    # print the labels
    currentClass = 0
    for i in range(0, 1400, 28):
        currentLabels = kmeans_model.labels_[i:i + 28]
        currentClass = currentClass + 1

def CreteDirectory(clusterID):
    file_path = "./" + str(clusterID) + "/"
    directory = os.path.dirname(file_path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    return file_path

if __name__ == "__main__":
    # features = ReadJSON2()
    # reduced_data = PCA(n_components=3).fit_transform(features)
    # # x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    # # y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
    #
    # #Plot2D(reduced_data)
    # Plot3D(reduced_data)

    #split the data as train and test
    (train, test, namesOfClasses, idsOfClasses, pathsOfTrain, pathsOfTest) = SplitData()

    #Print class names and ids
    for (names, ids) in zip(namesOfClasses, idsOfClasses):
        print("{}:{}".format(names, ids))

    #Create an np array from train data
    train_np = np.array(train, ndmin=2).reshape((1400, 2048))
    test_np = np.array(test, ndmin=2).reshape((600, 2048))

    #Run kmeans algorithm
    kmeans_model = KMeans(n_clusters=50, random_state=10, max_iter=300).fit(train_np)

    #Extract the labels
    labels = ExtractLabelsForClasses(kmeans_model)
    #print the labels
    for (i, label, id) in zip(range(0,1400, 28), labels, idsOfClasses):
        #print(kmeans_model.labels_[i:i + 28])
        print(label)
        print id

    classCounter = 0
    for (i, train) in zip(range(0,1400, 28), pathsOfTrain):
        labels = kmeans_model.labels_[i:i + 28]
        paths = pathsOfTrain[classCounter]
        for (label, path) in zip(labels, paths):
            destPath = CreteDirectory(label)
            copyfile(path, destPath + os.path.basename(path))
        classCounter = classCounter + 1



    #Test the model
    accuracy = TestModel(kmeans_model, test_np, pathsOfTest)
    print("Accuracy: {}".format(accuracy))


    a = 1
