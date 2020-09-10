import re
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

''' following defination is for n-grams where string is formated into n tuples '''
def ngrams_m(text ,n=3):

    remove=string.punctuation
    remove=remove.replace("#", "")
    pattern=r"[{}]".format(remove)
    text=re.sub(pattern ,r'',text)
    ngrams=zip(*[text[i:]for i in range(n)])
    return[''.join(ngram)  for ngram in ngrams]
count_vect=CountVectorizer(analyzer=ngrams_m)
tfidf_transformer=TfidfTransformer()
le =LabelEncoder()

''' This is the cosine similarity model for percentage of similarity'''
def cosinesim(text1,text2):

    if isinstance(text2,list):
        cos=dict()
        for i  in range(len(text2)):
            Targettxt=text2[i].lower()
            Sourcetxt=text1.lower()
            vect1=[Sourcetxt,Targettxt]
            vect2=count_vect.fit_transform(vect1)
            tfidf=tfidf_transformer.fit_transform(vect2)
            similarity=(tfidf * tfidf.T).A[0,1]
            cos.update({text2[i]:similarity})
        return [max(cos,key=cos.get),int(round(max(cos.values())* 100))]
    elif isinstance(text2,str):
        Targettxt="".join([item.lower() for item in text2])
        Sourcetxt="".join([item.lower() for item in text1])
        vect1=[Sourcetxt,Targettxt]
        vect2=count_vect.fit_transform(vect1)
        tfidf=tfidf_transformer.fit_transform(vect2)
        similarity=(tfidf * tfidf.T).A[0,1]
        return int(round(similarity * 100))
class fitmdl:
        """training the model and getting the  prdiction"""
        def __init__(self, x,y):
            self.x=x
            self.y=y
            self.Encode=le.fit(y)
            self.encode=le.transform(y)
            self.x_train_counts=count_vect.fit(x)
            self.X_train_counts=count_vect.transform(x)
            self.x_train_tfidf=tfidf_transformer.fit(self.X_train_counts)
            self.X_train_tfidf=tfidf_transformer.transform(self.X_train_counts)

        def model(self):
            knn = KNeighborsClassifier(n_neighbors=1,metric='cosine')
            clf=knn.fit(self.X_train_tfidf,self.encode)
            return clf

