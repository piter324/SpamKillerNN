from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup as bs

def get_mail_info(content):
    FROM_field = ""
    SUBJECT_field = ""
    if "Subject: " in content:
        SUBJECT_field = content[content.index("Subject: ") + len("Subject: "):-1]
        if "\n" in SUBJECT_field:
            SUBJECT_field = SUBJECT_field[0:SUBJECT_field.index("\n")]
        else:
            print("----Error-------- no substring!!!")
            print(content)
            print("End of error")
            SUBJECT_field = ""
    if "\nFrom: " in content:
        FROM_field = content[content.index("\nFrom: ") + len("\nFrom: "):-1]
        if "\n" in FROM_field:
            FROM_field = FROM_field[0:FROM_field.index("\n")]
        else:
            print("----Error-------- no substring!!!")
            print(content)
            print("End of error")
            FROM_field = ""
    return (FROM_field, SUBJECT_field)

def need_html_parsing(msg):
    tags=["a","abbr","acronym","address","area","b","base","bdo","big","blockquote","body","br","button","caption","cite","code","col","colgroup","dd","del","dfn","div","dl","DOCTYPE","dt","em","fieldset","form","h1","h2","h3","h4","h5","h6","head","html","hr","i","img","input","ins","kbd","label","legend","li","link","map","meta","noscript","object","ol","optgroup","option","p","param","pre","q","samp","script","select","small","span","strong","style","sub","sup","table","tbody","td","textarea","tfoot","th","thead","title","tr","tt","ul","var"]
    tags =["<"+x+">" for x in tags ]
    for i in tags:
        if i in msg:
            return True
    return False

def parse_html(msg):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(msg, 'lxml')
    text = soup.get_text()
    return text

def no_commas(text):
    return text.replace(",", "")

#data format: (is_spam ('0' or '1'), from, subject, is_formatted ('0' or '1'), message)
def get_data_for_csv(content, is_spam):
    if "\n\n" in content:
        start_of_mail = content.index("\n\n")
        header = content[0:start_of_mail]
        message = content[start_of_mail+1:-1]
        from_field, subj_field = get_mail_info(header)
        from_field = no_commas(from_field)
        subj_field = no_commas(subj_field)
        is_formatted = '0'
        if need_html_parsing(message):
            message = parse_html(message)
            is_formatted = '1'
        message = no_commas(message)
        if from_field is not "": 
             from_field = "\"" + from_field + "\""
        if subj_field is not "": 
             subj_field = "\"" + subj_field + "\""
        if message is not "": 
             message = "\"" + message + "\""
        return (is_spam,  from_field, subj_field, is_formatted, message)       
    else:
        return None

def get_file_content(filename):
    content = ""
    try:
        fh = open(filename, "r")
        content = fh.read()
        fh.close()
    except Exception as e:
        print("EEEEEEEEEEEEE")
        print(e)
        print(filename)
        print(">>>EEEEEEEEEEEEE")
    finally:
        return content

def create_csv(csv_name, path, is_spam):
    files = [f for f in listdir(path) if isfile(join(path, f)) ]
    files = files[0:len(files)-1]
    last_file = files[len(files)-1]
    counter = 0

    fh = open(csv_name, "a")
    for f in files:
        content = get_file_content(path+f)
        row = get_data_for_csv(content, is_spam)
        if row is not None and row[1] is not "" and row[2] is not "" and row[4] is not "":
            #write to csv file
            to_write = ",".join(row)
            if f is not last_file:
                to_write = to_write + "\n"
            counter = counter + 1
            try:
                fh.write(to_write)
            except Exception as e:
                print("---Exc in writing ----")
                print(e)
                print(to_write)
                print("-----------------------")
                counter = counter - 1

    fh.close()
    
    print("counter = ")
    print(counter)


if __name__ == "__main__":
    mypath_ham = "../ham/easy_ham/"
    mypath_spam = "../spam/spam_2/"
    create_csv("csv01.txt", mypath_ham, '0')
    create_csv("csv01.txt", mypath_spam, '1')





'''
filename = "../spam/spam_2/00002.9438920e9a55591b18e60d1ed37d992b"
mypath_ham = "../ham/easy_ham/"
mypath_spam = "../spam/spam_2/"

fh = open(filename, "r")
content = fh.read()
fh.close()

#print(content)
print("yaay")

#listing files
ham_files = [f for f in listdir(mypath_ham) if isfile(join(mypath_ham, f))]
spam_files = [f for f in listdir(mypath_spam) if isfile(join(mypath_spam, f)) ]
#deleting cmds files
ham_files = ham_files[0:len(ham_files)-1]
spam_files = ham_files[0:len(spam_files)-1]

a = content.split("\n\n") #not really good, because there may be double n in the message
is_mail_formatted = '0'
part1 = "lalalala"
part2 = "lalalalala"

if "\n\n" in content:
    b = content.index("\n\n")
    part1 = content[0:b]
    part2 = content[b+1:-1]
    print(part1)
    print("-----------------------------")
    to, subj = get_mail_info(part1)
    print("TO ---------")
    print(to)
    print("SUBJ--------")
    print(subj)
    if need_html_parsing(part2):
        print("------------------------uhuhu-----------------")
        txt = parse_html(part2)
        print(txt)
#jeżeli nie ma, możemy się zmartwić :(


'''


