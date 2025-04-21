import scipy.io as scipy
import Utils




mat = scipy.loadmat('grupoDados1.mat')

grupoTest = mat['grupoTest']
grupoTrain = mat['grupoTrain']
testRots = mat['testRots']
trainRots = mat['trainRots']

print("Para k=1")
accuracy = Utils.accuracy(grupoTrain, trainRots, grupoTest, testRots, 1)
print(f"Acurácia: {accuracy}\n")

print("Para k=10")
accuracy = Utils.accuracy(grupoTrain, trainRots, grupoTest, testRots, 10)
print(f"Acurácia: {accuracy}\n")