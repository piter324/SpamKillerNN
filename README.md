# SpamKillerNN
This repo now contains only a basic implementation of a neural network found online.

## Bag of words
The file "bag_of_words.csv" contains 2000 most common words from e-mails in "csv01_properUTF8.txt" file. They are ordered by the number of occurences descending, which means the most common word is in the first row.

The structure:
`<word> <number of occurences in all emails>`
one word per line

## CSV with number of occurences
A file "mails_with_word_occurences.csv" contains information about whether the message is SPAM or HAM and the number of occurences of each of the most common words in every e-mail message.

The structure:

`<1 for SPAM, 0 for HAM> <occurences of the first word> <occurences of the second word> <occurences of the third word> ... <occurences of the 2000th word>`

One line represents one message in the same order as in "csv_properUTF8.txt" file.