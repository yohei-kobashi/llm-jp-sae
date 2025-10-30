import csv

data_1 = [["sentence,tense,aspect"]]
data_2 = [["sentence,entailment"]]
saved_sentences = set([])
for i,(sentence_1,sentence_2,label,tense_sentence_1_tense_sentence_2,aspect_sentence_1_aspect_sentence_2) in enumerate(csv.reader(open("data/TEA.txt"))):
    if not i: continue
    tenses = tense_sentence_1_tense_sentence_2.split(" - ")
    aspects = aspect_sentence_1_aspect_sentence_2.split(" - ")
    if not sentence_1 in saved_sentences:
        data_1.append([sentence_1, tenses[0], aspects[0]])
    if not sentence_2 in saved_sentences:
        data_1.append([sentence_2, tenses[1], aspects[1]])
    saved_sentences.update([sentence_1, sentence_2])
    data_2.append([f'"{sentence_1}" entails "{sentence_2}"', label])

csv.writer(open("data/tense_aspect_list.csv", "w")).writerows(data_1)
csv.writer(open("data/TEA_list.csv", "w")).writerows(data_2)