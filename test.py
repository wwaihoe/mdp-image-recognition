from classification import classify

import os


path = os.getcwd()
os.chdir(path)


path = "test.jpg"
pred = classify(path)

print(pred)