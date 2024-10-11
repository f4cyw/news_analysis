import nltk
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import itertools
import numpy as np
import platform
import matplotlib.font_manager as fm
import matplotlib as mpl
import re

nltk.download('punkt')

try:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('korean'))
except OSError:
    stop_words = set()

# Set font to avoid issues with Korean text
def set_font():
    if platform.system() == "Darwin":  # macOS
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    elif platform.system() == "Windows":  # Windows
        font_path = "C:\Windows\Fonts\malgun.ttf"
    else:  # Other (Linux, etc.)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rc('font', family=font_prop.get_name())
    mpl.rcParams['axes.unicode_minus'] = False  # To properly display negative signs
    return font_prop

def remove_particles(word):
    # Remove common Korean particles (조사) using regex
    return re.sub(r'(은|는|이|가|을|를|에|의|와|과|도|로|으로|에서|에게|한테|께|까지|만|조차|마저|부터|밖에)$', '', word)

def extract_keywords(text, num_keywords=10):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords, non-alphabetic tokens, and Korean particles
    keywords = [remove_particles(word) for word in tokens if word.isalpha() and word not in stop_words]
    keywords = [word for word in keywords if word]  # Remove empty strings after particle removal
    
    # Calculate TF-IDF scores
    vectorizer = CountVectorizer()
    word_counts = vectorizer.fit_transform([' '.join(keywords)])
    tfidf = TfidfTransformer().fit_transform(word_counts)
    scores = tfidf.toarray()[0]
    
    # Get the top keywords
    keyword_scores = list(zip(vectorizer.get_feature_names_out(), scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    top_keywords = [keyword for keyword, score in keyword_scores[:num_keywords]]
    
    return top_keywords, keyword_scores

def build_relationship_graph(keywords, keyword_scores, additional_keywords=None):
    # Add additional keywords to the list of keywords if provided
    if additional_keywords:
        keywords += additional_keywords
    keywords = list(set(keywords))  # Remove duplicates
    
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes and weighted edges based on TF-IDF scores
    G.add_nodes_from(keywords)
    for combination in itertools.combinations(keywords, 2):
        weight = np.mean([dict(keyword_scores).get(combination[0], 0), dict(keyword_scores).get(combination[1], 0)])
        G.add_edge(*combination, weight=weight)
    
    return G

def visualize_graph(G):
    # Set font for graph visualization
    font_prop = set_font()
    
    # Draw the graph
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    nx.draw_networkx(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_size=10, font_weight='bold', width=weights, font_family=font_prop.get_name())
    plt.show()

def main(article, additional_keywords=None):
    set_font()  # Set the appropriate font for the system
    keywords, keyword_scores = extract_keywords(article)
    G = build_relationship_graph(keywords, keyword_scores, additional_keywords)
    visualize_graph(G)

if __name__ == "__main__":
    articles =  """
    그룹 아일릿이 뉴진스를 표절했다는 의혹에 다시 불이 붙었다. 민희진 어도어 전 대표 측이 하이브 내부 직원으로부터 제보를 받았다고 주장하고 나선 것.

오늘(11일) 서울중앙지방법원 민사합의50부는 민 전 대표의 대표이사직 복귀를 요구하는 가처분 신청에 대한 심문을 진행했다.

이날 민 전 대표 측은 하이브 내부 직원과의 대화 내용을 공개했다.

이 내부 직원은 아일릿 크리에이티브 디렉터가 아일릿 구상 단계부터 뉴진스의 기획안을 요청했다고 말했다. 또한 아일릿의 기획안이 뉴진스의 기획안과 똑같다고 주장했다.

대화 내용 중에는 이 직원이 "똑같이 만들 거라고는 정말 상상도 못 했다"고 말하는 대목도 있다. 그는 아일릿의 소속사인 빌리프랩에서 표절 의혹을 부인하는 것에 대해 불편함을 느낀다고도 했다.

민 전 대표 측은 이 대화 내용을 토대로 "빌리프랩은 표절 의혹이 사실이 아니라고 지속적으로 부인하고 하이브는 이를 방치했다"며, 지난 4월 하이브의 감사가 불법이라고 강조했다.
    """
    main(articles, additional_keywords=["민희진"])