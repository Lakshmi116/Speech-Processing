import os


nos = 101

words = []

root_dir = "./home/gdata/narayana/Lakshmi/"
data_dir = os.path.join(root_dir, "Data")

file = open(root_dir + "speaker_wordlist.csv", "r")
lines = file.readlines()[1:nos+1]
for line in lines:
    label = line.split(',')[0].strip()
    words.append(label)

words = list(set(words))

file = open(data_dir + f"classmap{nos}.csv", "w")
file.write("Label,Class\n")
for idx, word in enumerate(words):
    line = f"{word},{idx}\n"
    file.write(line)
    print(line, end="")


exit()

