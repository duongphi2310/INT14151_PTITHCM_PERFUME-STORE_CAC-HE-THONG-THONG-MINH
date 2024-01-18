import json
import re
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.neighbors import NearestNeighbors
import warnings
import pymysql

warnings.filterwarnings("ignore")

# Thay thế các giá trị dưới đây bằng thông tin kết nối của bạn
host = "127.0.0.1"
user = "root"
password = ""
database = "perfume_shop"

# Kết nối đến cơ sở dữ liệu MySQL
connection = pymysql.connect(host=host, user=user, password=password, database=database, charset='utf8mb4')

# Tạo một con trỏ để thực hiện các truy vấn SQL
cursor = connection.cursor()

# Thực hiện truy vấn SQL để lấy dữ liệu từ MySQL (bao gồm cả các trường "Name", "Description" và "Style")
query = "SELECT Name, Description, Style FROM products"
cursor.execute(query)

# Lấy kết quả của truy vấn
results = cursor.fetchall()

# Tạo DataFrame từ kết quả truy vấn
columns = ["Name", "Description", "Style"]
data = [list(row) for row in results]
perfumes = pd.DataFrame(data, columns=columns)

# Đóng con trỏ và kết nối MySQL
cursor.close()
connection.close()

def preprocess_text(text):
    vietnamese_characters = "a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễđìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ"
    pattern = f"[^{vietnamese_characters}' ]"
    cleaned_text = re.sub(pattern, '', text).lower()
    return cleaned_text

perfumes['Description'] = perfumes['Description'] + perfumes['Style']
perfumes['Description'] = perfumes['Description'].apply(preprocess_text)

po = PorterStemmer()
# cv = CountVectorizer(stop_words='english')
# clean_description = cv.fit_transform(perfumes['Description']).toarray()
stop_words_vietnamese = ['là', 'đó', 'thì', 'của', 'và', 'ở', 'có', 'trong', 'theo', 'nhưng']
cv = CountVectorizer(stop_words=stop_words_vietnamese)
clean_description = cv.fit_transform(perfumes['Description']).toarray()

model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(clean_description)

def find_similar_perfumes(user_description):
    clean_user_description = cv.transform([user_description]).toarray()
    distances, indices = model.kneighbors(clean_user_description, n_neighbors=6)
    similar_perfumes = list(perfumes['Name'][indices[0][:]].values)
    return similar_perfumes

if __name__ == "__main__":
    # user_description = sys.argv[1]
    user_description = "Thanh lich, nu tinh, quyen ru"

    similar_perfumes = find_similar_perfumes(user_description)
    print(json.dumps({'similar_perfumes': similar_perfumes}))
    # print(({'similar_perfumes': similar_perfumes}))
