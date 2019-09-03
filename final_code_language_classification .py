import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score
"""*******************************************************************************"""

file = r'/home/snehasis/Desktop/NLP PROJECT/CODE/final_dataset.csv'
data = pd.read_csv(file)
col1 = []
col1 = data['SENTENCES'].str.lower()


# tokenize sentences
for i in range(1545):
    tokenizer = RegexpTokenizer('\w+|,\S')
    if type(col1[i]) is str:
        col1[i] = tokenizer.tokenize(col1[i])

"""**********************************************************************************"""


# tokenized and labeled sentences
df = pd.DataFrame({'SENTENCES': col1, 'LABELS': data['LABELS']})

list_hindi = []    # contains hindi tokenized sentences
list_eng = []      # contains english tokenized sentences

c_hindi = 0
# count and append sentences to list_hind[]
for i in range(1545):
    if df.iloc[i, 1] == 'Hindi':
        c_hindi += 1
        list_hindi.append(df.iloc[i, 0])

c_eng = 0
# count and append sentences to list_eng[]
for i in range(1545):
    if df.iloc[i, 1] == 'English':
        c_eng += 1
        list_eng.append(df.iloc[i, 0])

list_hindi_words = []
c1 = 0
for i in range(len(list_hindi)):
    for j in range(len(list_hindi[i])):
        c1 += 1
        list_hindi_words.append(list_hindi[i][j])

list_eng_words = []
c2 = 0
for i in range(len(list_eng)):
    for j in range(len(list_eng[i])):
        c2 += 1
        list_eng_words.append(list_eng[i][j])

total_hi_words = len(list_hindi_words)
total_en_words = len(list_eng_words)

"""*****************************************************************************************"""

"""frequency distribution of each language words"""

fd_eng = nltk.FreqDist(list_eng_words)
unique_en_words_num = len(fd_eng)  # num of unique eng words

fd_hindi = nltk.FreqDist(list_hindi_words)
unique_hi_words_num = len(fd_hindi)  # num of unique hi words

"""*************************************************************************"""
""" Prior probabilities of classes """
hindi_sent_num = len(list_hindi)
eng_sent_num = len(list_eng)
total_sent_num = hindi_sent_num + eng_sent_num

hi_prior_prob = hindi_sent_num / total_sent_num
en_prior_prob = eng_sent_num / total_sent_num


"""***************************************************************************"""

# a function to do smoothing


def smoothed_probability_of_words(fd, tot_fq, tot_words, w):
            fq = fd.get(w)
            if fq is not None:
                return (fq + 1) / (tot_words + tot_fq)
            elif fq is None:
                return (0 + 1) / (tot_words + tot_fq)


# classify the sentence
def classify(list_hi, list_en, list_len):
                hi_prob = 1
                for itr in range(list_len):
                    hi_prob *= list_hi[itr]
                hi_prob *= hi_prior_prob

                en_prob = 1
                for itr in range(list_len):
                    en_prob *= list_en[itr]
                en_prob *= en_prior_prob
                if hi_prob > en_prob:
                    return 1
                else:
                    return 0


# labels the sentence
def label_class(pr_list_hi, pr_list_en, pr_list_len):
                class_ = classify(pr_list_hi, pr_list_en, pr_list_len)
                if class_ == 1:
                    # print('Hindi')
                    return 'Hindi'
                elif class_ == 0:
                    # print('English')
                    return 'English'


"""---------------------------------------------------------------------------------------------------------"""
"""   to test data    """
file1 = r'/home/snehasis/Desktop/NLP PROJECT/CODE/test_data1.csv'
test_data = pd.read_csv(file1)

col = test_data['SENTENCES'].str.lower()
# tokenize sentences
for i in range(len(col)):
    tokenizer = RegexpTokenizer('\w+|,\S')
    if type(col[i]) is str:
        col[i] = tokenizer.tokenize(col[i])

true_labels = test_data['LABELS']
# stores the predicted labels of each sentence
predicted_labels = []
for i in range(len(col)):
    prob_list_hi = []
    prob_list_en = []
    test_sent = col[i]
    if type(test_sent) is list:
        for j in range(len(test_sent)):
            word = test_sent[j]
            prob_list_hi.append(smoothed_probability_of_words(fd_hindi, unique_hi_words_num, total_hi_words, word))
            prob_list_en.append(smoothed_probability_of_words(fd_eng, unique_en_words_num, total_en_words, word))
        label = label_class(prob_list_hi, prob_list_en, len(test_sent))
        predicted_labels.append(label)


# Calculate accuracy score of the test set
accuracy = accuracy_score(true_labels, predicted_labels)
print("accuracy score : ", accuracy)













