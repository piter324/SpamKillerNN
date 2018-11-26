Wersja csv01 zawiera:
*2497 rekordów, które nie s¹ spamem z pliku 20030228_easy_ham.tar.bz2
*1290 rekordów, które s¹ spamem z pliku 20050311_spam_2.tar.bz2
Czêœæ mi siê nie wczyta³a - sprawdzê to póŸniej dlaczego dok³adnie. Mo¿na by te¿ dorzuciæ w kolejnych wersjach pliki z reszty.

Format danych:
is_spam, from, subject, is_formatted, message
*Po ka¿dym rekordzie poza ostatnim jest \n, ale mo¿e te¿ byæ on w œrodku wiadomoœci
*is_spam i is_formatted przyjmuje wartoœci '0' i '1' (fa³sz i prawda) - jako teskt!
*from, subject i message s¹ w cudzys³owiach "

Kod importdata.py nie jest jeszcze ostateczn¹ wersj¹ tego kodu.