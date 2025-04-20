import scipy.io as scipy
import Utils.meuKnn as meuKnn




mat = scipy.loadmat('grupoDados1.mat')

grupoTest = mat['grupoTest']
grupoTrain = mat['grupoTrain']
testRots = mat['testRots']
trainRots = mat['trainRots']
print(grupoTest)
print(grupoTrain)
print(testRots)
print(trainRots)


meuKnn(grupoTrain, testRots, grupoTest, 3)