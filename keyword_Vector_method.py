import nltk
import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models import Phrases
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import itertools
import numpy as np
import platform
import matplotlib.font_manager as fm
import matplotlib as mpl
import fasttext

nltk.download('punkt')

try:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('korean')) | {'때문', '이유', '말', '것', '등', '및', '수', '년', '월', '일', '때', '중', '한'}
except OSError:
    stop_words = set()

def set_font():
    if platform.system() == "Darwin": 
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    elif platform.system() == "Windows":  
        font_path = "C:\\Windows\\Fonts\\malgun.ttf"
    else: 
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rc('font', family=font_prop.get_name())
    mpl.rcParams['axes.unicode_minus'] = False 
    return font_prop

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    tokens = [re.findall(r'[가-힣]{2,}', sentence) for sentence in sentences]  # Extract words with two or more Korean characters
    keywords = [[word for word in sentence if word not in stop_words] for sentence in tokens]
    return [sentence for sentence in keywords if len(sentence) > 0]

def extract_top_keywords(text, num_keywords=10):
    tokens = re.findall(r'[가-힣]{2,}', text)
    tokens = [word for word in tokens if word not in stop_words]
    if not tokens:
        tokens = ['기술', '주식', '경제']  # Default keywords if filtering results in an empty list

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([' '.join(tokens)])
    scores = tfidf.idf_
    keyword_scores = list(zip(tfidf.get_feature_names_out(), scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    top_keywords = [keyword for keyword, score in keyword_scores[:num_keywords]]
    return top_keywords

def build_relationship_graph(model, keywords, additional_keywords=None):
    if additional_keywords:
        keywords += additional_keywords
    keywords = list(set(keywords)) 
    
    G = nx.Graph()
    G.add_nodes_from(keywords)
    for combination in itertools.combinations(keywords, 2):
        vector1 = model.get_word_vector(combination[0])
        vector2 = model.get_word_vector(combination[1])
        if vector1 is not None and vector2 is not None:
            # Calculate cosine similarity
            similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            if similarity > 0.1:  # Threshold for similarity
                G.add_edge(combination[0], combination[1], weight=similarity)
    
    return G

def visualize_graph(G):
    font_prop = set_font()
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    edge_colors = [plt.cm.plasma(weight) for weight in weights]
    edge_widths = [weight * 2 for weight in weights]  # Scale edge widths based on similarity
    
    plt.figure(figsize=(8, 8))
    nx.draw_networkx(G, pos, with_labels=True, node_size=1500, node_color='skyblue', 
                     font_size=10, font_weight='bold', width=edge_widths, edge_color=edge_colors, 
                     edge_cmap=plt.cm.plasma, alpha=0.8, font_family=font_prop.get_name())
    plt.title("Keyword Relationship Graph (FastText)", fontsize=16)
    plt.show()

def main(article, additional_keywords=None):
    set_font()
    sentences = preprocess_text(article)
    model = fasttext.load_model("model/facebook_ko.bin")
    keywords = extract_top_keywords(article)  # Extract top keywords using TF-IDF
    G = build_relationship_graph(model, keywords, additional_keywords)
    visualize_graph(G)

if __name__ == "__main__":
    articles = """기술주를 중심으로 미국 뉴욕증시가 지지부진한 흐름을 나타내고 있다. ‘인공지능(AI) 거품론’ ‘반도체 고점론’ 등 비관론이 퍼지면서 전체 증시가 한 차례 출렁인 가운데 11월 대선, 연방준비제도(연준)의 금리인하 속도 등 불확실한 요인들이 하방 압력을 가하고 있기 때문이다. 장우석 유에스스탁 부사장은 10월 8일 인터뷰에서 “이럴 때 엔비디아를 사 모아야 한다”고 말했다. 최근 서학개미는 기존에 선호하던 기술주를 대거 처분하고 배당주, 현금 등 안전자산 쪽으로 투자 노선을 갈아탄 상태인데, 이에 대해 “지금은 떠날 타이밍이 아니다”라고 조언한 것이다. 다음은 장 부사장과 나눈 일문일답.
    “美 경기침체, 처음부터 없었다”
    11월 대선까지는 증시가 횡보할 것이라는 전망이 많은데.
    ‌
    “원래 대선 직전에는 증시가 재미없다. 한 달간 옆으로 긴 장이 이어지다가 선거가 끝난 뒤부터 오를 것이다. 또 지금은 누가 대통령이 되느냐보다 금리가 얼마나 떨어질지가 더 관건이라서 11월 5일(현지 시간) 대선이 끝나더라도 경우에 따라 그 직후 상승세가 강하지 않을 수 있다.”
    ‌
    고용지표 강세로 11월 연준이 금리를 동결할 수 있다는 분석도 일각에서 나오는데.
    ‌
    “9월 고용지표가 예상을 뛰어넘으면서 ‘금리인하를 괜히 한 것 아니냐’는 반응이 나오고 있다. 다만 개인적으로는 2022년 6월 9.1% 상승률을 보인 소비자물가지수(CPI)가 지금 2.5%까지 내려왔기 때문에 인하는 당연한 수순 아닌가 싶다. 이번 주 발표 예정인 9월 CPI가 전망치(2.3%)에서 크게 어긋나지만 않으면 아마 0.25%p 인하는 할 거라고 본다. 아직까지 동결 전망은 소수다.”
    ‌
    이번 고용지표로 경기침체 우려는 어느 정도 불식됐다고 보면 되나.
    ‌
    “침체는 없다. 분기 국내총생산(GDP) 성장률이 3%대인 국가에 침체는 애초에 어울리지 않는 얘기다. 7월 미국 공급관리협회(ISM) 제조업지수가 큰 폭으로 떨어지면서 갑자기 위기감이 고조됐던 것인데, 지금은 흔적도 없이 사라져가고 있다. 어제도 글로벌 투자은행(IB) 골드만삭스가 미국 경기침체 확률을 15%로 낮춰 잡았다. 15%면 ‘없다’는 뜻으로 봐도 무방하다.”
    """
    main(articles, additional_keywords=[])
