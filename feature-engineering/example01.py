from functools import reduce
import numpy as np

texts = [['i', 'have', 'a', 'cat'], 
        ['he', 'have', 'a', 'dog'], 
        ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

dictionary = list(enumerate(set(reduce(lambda x, y: x+y, texts))))  

def bag_of_words(sentence):
    vector = np.zeros(len(dictionary))

    for i, word in dictionary:
        count = 0
        for w in sentence:
            if w == word:
                count+=1
        vector[i] = count

    return vector 

for i in texts:
    print(bag_of_words(i))


print("################################################")
################################################


from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import euclidean

vect = CountVectorizer(ngram_range= (1, 1))
vect.fit_transform(['you have no dog', 'no, you have dog']).toarray()

print(vect.vocabulary_)

vect = CountVectorizer(ngram_range= (1, 2))
vect.fit_transform(['you have no dog', 'no, you have dog']).toarray()

print(vect.vocabulary_)

### analyzer = 'char_wb' --> Tokenize theo character 
vect = CountVectorizer(ngram_range = (3, 3), analyzer = 'char_wb')
n1, n2, n3, n4 = vect.fit_transform(['andersen', 'peterson', 'petrov', 'smith']).toarray()
euclidean(n1, n2), euclidean(n2, n3), euclidean(n3, n4)


print("################################################")
################################################


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
 	'tôi thích ăn bánh mì nhân thịt',
	'cô ấy thích ăn bánh mì, còn tôi thích ăn xôi',
	'thị trường chứng khoán giảm làm tôi lo lắng',
	'chứng khoán sẽ phục hồi vào thời gian tới. danh mục của tôi sẽ tăng trở lại',
  'dự báo thời tiết hà nội có mưa vào chiều và tối. tôi sẽ mang ô khi ra ngoài'
]

# Khởi tạo model tính tfidf cho mỗi từ
# Tham số max_df để loại bỏ các từ stopwords xuất hiện ở hơn 90% các câu.
vectorizer = TfidfVectorizer(max_df = 0.9)

# Tokenize các câu theo tfidf
X = vectorizer.fit_transform(corpus)

print('words in dictionary:')
print(vectorizer.get_feature_names())
print('X shape: ', X.shape)

# Ta có thể thấy từ tôi xuất hiện ở toàn bộ các câu và không mang nhiều 
# ý nghĩa của chủ đề của câu nên có thể coi là một stopword. 
# Bằng phương pháp lọc cận trên của tần suất xuất hiện từ trong 
# văn bản là 90% ta đã loại bỏ được từ này khỏi dictionary.