#Document Similarity 2 - Python - Jose Ahirton Lopes (FCamara)

"""
__author__ = "Ahirton Lopes"
__copyright__ = "Copyright 2017, FCamara/Duratex"
__credits__ = ["Ahirton Lopes"]
__license__ = "None"
__version__ = "1.0"
__maintainer__ = "Ahirton Lopes"
__email__ = "ahirtonlopes@gmail.com"
__status__ = "Beta"
"""

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt') #Dicionários para reconhecimento de pontuação

#Stemmização de documentos // Dominuição das palavras ao seu radical tendo em vista facilitação de uso da técnica TF-IDF
stemmer =  nltk.stem.RSLPStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

#Transformação em minúsculas, remoção de pontuação e tokenização em português
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map), language='portuguese'))

my_stopword_list = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j',
          'k','l','ç','z','x','c','v','b','n','m','Q','W','E','R','T','Y','U',
          'I','O','P','A','S','D','F','G','H','J','K','L','Ç','Z','X','C','V',
          'B','N','M','!','@','#','$','%','¨','&','*','(',')','_','+','-','--','=',
          '´','`','^','~',':',';','?','|','{','[','}','<','>','.',',','/','//','...',
          '"',"'","''",'``','no', 'na', 'do', 'da', 'de', 'as', 'os', 'nos', 'nas', 
          'dos', 'das', 'se', 'em','para','que','pela','pelo', 'com','sem', 'c/', 's/',
          'um','uma','pra',' ', 'aos', 'etc', 'e/ou', 'ou','ate','por','como', 'ao',
          'nao','mais','maior','menor','tambem', 'ja',
          'ele','ela','aquilo','aquele','aquela','isso','esse','essa','este','esta',
          'sua','seu', 'neste', 'nesta', 'nesse', 'nessa',
          'algum','alguma','alguns','algumas', 'porque','por que', 'nem', 'rt', 'me', 'http', 'https']

#Vetorização dos documentos já normalizados tendo em vista "my_stopword_list"
vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=my_stopword_list)

#Verificação de termos relevantes e comparação via técnica TF-IDF dos documentos vetorizados
def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]
    
#"Frases alvo" a serem analisados junto a documentos

print cosine_sim('Venho por meio desta idealizar uma pequena melhoria na Ducha Hydra Star music. Na Ideia anterior que aqui formalizei, era economizar a energia eletrica com aviso atraves de voz que o banho estaria terminando.  Nesta nova Ideia que tive, e que a Ducha pudesse atender a varios usuarios com programacao de temperatura, tempo de banho e opcoes de voz ( masculina e feminina ) para avisar quando o tempo de banho programado, estivesse terminando. ', 'Venho por meio desta idealizar uma pequena melhoria na Ducha Hydra Star music. Na Ideia anterior que aqui formalizei, era economizar a energia eletrica com aviso atraves de voz que o banho estaria terminando.  Nesta nova Ideia que tive, e que a Ducha pudesse atender a varios usuarios com programacao de temperatura, tempo de banho e opcoes de voz ( masculina e feminina ) para avisar quando o tempo de banho programado, estivesse terminando. ')
print cosine_sim('Venho por meio desta idealizar uma pequena melhoria na Ducha Hydra Star music. Na Ideia anterior que aqui formalizei, era economizar a energia eletrica com aviso atraves de voz que o banho estaria terminando.  Nesta nova Ideia que tive, e que a Ducha pudesse atender a varios usuarios com programacao de temperatura, tempo de banho e opcoes de voz ( masculina e feminina ) para avisar quando o tempo de banho programado, estivesse terminando. ', 'criacao de um kit para banheira com sistema hi fi e wireless, com um programa para instalacao no celular. onde possibilida ao comprador ter acesso via internet do celular a uma programacao de um banho de sua preferencia. este programa teria que possibilar o acesso a quantidade de agua, temperatura, espuma, e outras preferencias. possibilitando assim que ao sair de um compromisso profissional ou de lazer a pessoa nao precisse chegar em casa esperar ate que apronte seu banho, ele ja vai estar pronto e so chegar e relaxar.')
print cosine_sim('CHUVEIRO ELETRONICO COM SENSOR TERMOSTATO OU TERMOMETRO QUE REGULEA TEMPERATURA DA aGUA CONFORME A TEMPERATURA MINIMA OU MAXIMA DO DIA EM CURSO', 'CHUVEIRO ELETRONICO COM SENSOR TERMOSTATO OU TERMOMETRO QUE REGULEA TEMPERATURA DA aGUA CONFORME A TEMPERATURA MINIMA OU MAXIMA DO DIA EM CURSO')
print cosine_sim('CHUVEIRO ELETRONICO COM SENSOR TERMOSTATO OU TERMOMETRO QUE REGULEA TEMPERATURA DA aGUA CONFORME A TEMPERATURA MINIMA OU MAXIMA DO DIA EM CURSO', 'CHUVEIRO ELETRONICO COM SENSOR TERMOSTATO OU TERMOMETRO QUE REGULEA TEMPERATURA DA aGUA CONFORME A TEMPERATURA MINIMA OU MAXIMA DO DIA EM CURSO')