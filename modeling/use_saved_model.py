from machine import BulletinMachine

def classifier_creator(indata, outdata, class_weight=None):
    return RandomForestClassifier(n_estimators=100, class_weight=class_weight)

def regressor_creator(indata, outdata):
    return MultiTaskElasticNet()

ubm = BulletinMachine.load("demo")



