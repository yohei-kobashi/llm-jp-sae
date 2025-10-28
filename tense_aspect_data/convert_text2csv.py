import csv

data = [["sentence,tense,aspect"]]
for i,(sentence_1,sentence_2,label,tense_sentence_1_tense_sentence_2,aspect_sentence_1_aspect_sentence_2) in enumerate(csv.reader(open("tense_aspect_data/TEA.txt"))):
    if not i: continue
    tenses = tense_sentence_1_tense_sentence_2.split(" - ")
    aspects = aspect_sentence_1_aspect_sentence_2.split(" - ")
    data.append([sentence_1, tenses[0], aspects[0]])
    data.append([sentence_2, tenses[1], aspects[1]])

csv.writer(open("tense_aspect_data/tense_aspect_list.csv", "w")).writerows(data)