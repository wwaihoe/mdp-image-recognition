from classification import classify

import os


path = "D:/OneDrive - Nanyang Technological University/23S2/MDP"
os.chdir(path)


path = "photos/original/Alphabet_A_10.jpg"
pred = classify(path)

print(pred)