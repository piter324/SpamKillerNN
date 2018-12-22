from collections import Counter
import sanitizedata as sd
import pprint
import csv

def combine_words_array(csv_arr):
    words_arr = []
    for arr in csv_arr:
        words = arr[2] + arr[4]
        words_arr.append(words)
    return words_arr

def save_bag_of_words(all_mails, bag_csv_filename, how_many):
    bag = Counter()
    for words_arr in all_mails:
        for word in words_arr:
            bag[word] += 1
    
    most_common = bag.most_common(how_many)
    with open(file=bag_csv_filename, mode='w', newline='') as csvfile:
        bagwriter = csv.writer(csvfile)
        for row in most_common:
            bagwriter.writerow([row[0], row[1]])
        
    # print(bag)
    # print(most_common)

def analyze_mails_for_most_common_words(most_common_csv_filename, all_mails, clear_mail_vectors, output_csv_filename):

    with open(file=most_common_csv_filename, mode='r') as inputcsv:
        inputrdr = csv.reader(inputcsv)
        most_common = []
        for row in inputrdr:
            most_common.append(row[0])
        occurences = []
        for idx, words_arr in enumerate(all_mails):
            this_mail_occurences = [clear_mail_vectors[idx][0]]
            for common in most_common:
                this_mail_occurences.append(words_arr.count(common))
            occurences.append(this_mail_occurences)
    
    with open(file=output_csv_filename, mode='w') as outputcsv:
        csvwriter = csv.writer(outputcsv)
        for occ in occurences:
            csvwriter.writerow(occ)
    

if __name__ == "__main__":
    test = False
    csvinput = "csv01.csv" if test else "csv01_properUTF8.txt"
    bagofwords = "bag_of_words_test.csv" if test else "bag_of_words.csv"
    outputcsv = "mails_with_word_occurences_test.csv" if test else "mails_with_word_occurences.csv"

    csv_arr = sd.prepare_csv(csvinput)
    all_mails = combine_words_array(csv_arr)

    # save_bag_of_words(all_mails, "bag_of_words_test.csv", 2000)
    analyze_mails_for_most_common_words(bagofwords, all_mails, csv_arr, outputcsv)
