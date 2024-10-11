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


def set_font():
    if platform.system() == "Darwin": 
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    elif platform.system() == "Windows":  
        font_path = "C:\Windows\Fonts\malgun.ttf"
    else: 
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rc('font', family=font_prop.get_name())
    mpl.rcParams['axes.unicode_minus'] = False 
    return font_prop

def remove_particles(word):
    
    return re.sub(r'(은|는|이|가|을|를|에|의|와|과|도|로|으로|에서|에게|한테|께|까지|만|조차|마저|부터|밖에)$', '', word)

def remove_unwanted_verbs(word):
    
    return re.sub(r'(것|하다|되다|이다|있다|없다|같다|되다|보다|싶다|만들다|보다|주다|받다|오다|가다|나다|살다)$', '', word)

def extract_keywords(text, num_keywords=10):
    
    tokens = nltk.word_tokenize(text)
    
    keywords = [remove_unwanted_verbs(remove_particles(word)) for word in tokens if word.isalpha() and word not in stop_words]
    keywords = [word for word in keywords if word] 
    
   
    vectorizer = CountVectorizer()
    word_counts = vectorizer.fit_transform([' '.join(keywords)])
    tfidf = TfidfTransformer().fit_transform(word_counts)
    scores = tfidf.toarray()[0]
    
   
    keyword_scores = list(zip(vectorizer.get_feature_names_out(), scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    top_keywords = [keyword for keyword, score in keyword_scores[:num_keywords]]
    
    return top_keywords, keyword_scores

def build_relationship_graph(keywords, keyword_scores, additional_keywords=None):
    
    if additional_keywords:
        keywords += additional_keywords
    keywords = list(set(keywords)) 
    
    
    G = nx.Graph()
    
    
    G.add_nodes_from(keywords)
    for combination in itertools.combinations(keywords, 2):
        weight = np.mean([dict(keyword_scores).get(combination[0], 0), dict(keyword_scores).get(combination[1], 0)])
        G.add_edge(*combination, weight=weight)
    
    return G

def visualize_graph(G):
    
    font_prop = set_font()
    
    
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    nx.draw_networkx(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_size=10, font_weight='bold', width=weights, font_family=font_prop.get_name())
    plt.show()

def main(article, additional_keywords=None):
    set_font()  
    keywords, keyword_scores = extract_keywords(article)
    G = build_relationship_graph(keywords, keyword_scores, additional_keywords)
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
‌
거시 불확실성 속에서 기업들의 3분기 실적이 주목받고 있다.
‌
“전체 기업의 3분기 평균 매출, 주당순이익(EPS) 상승률이 모두 10%대로 예상된다. 나쁘지 않은 편이고, 4분기에는 이보다 더 좋아질 것으로 보인다. 그러니 장이 지지부진하다고 해도 팔고 도망갈 게 아니라, 이럴 때 사야 한다는 생각이다. 제레미 시겔 미국 펜실베이니아대 와튼스쿨 교수가 최근 ‘미국 경제가 너무 좋고 기업들 실적도 계속 개선되고 있어 올해 S&P500 지수가 6000에 도달할 것’이라고 말했다. 주요 기업의 3분기 실적이 하나 둘 발표되면서 주가가 본격적으로 움직일 것 같다.
‌
가장 눈여겨볼 섹터는 IT(정보기술), 금융, 에너지다. 인플레이션이 잦아들 때 실적이 개선되는 부동산도 좋다. 최근 경기침체 우려가 커지면서 소매업종 실적이 저조할 것으로 예상되는데, 그것을 제외하고는 대체로 다 괜찮다.”

“젠슨 황, AI 산업 저변 확대 나서”
서학개미의 주된 관심사인 기술주 실적을 구체적으로 전망한다면.
‌
“3분기 S&P500 전체 실적에서 기술주가 기여한 비중이 지난해 같은 기간보다 30%가량 증가한 것으로 분석됐다. 모든 섹터를 통틀어 가장 높은 상승률이다. 매그니피센트7(M7: MS·애플·아마존·엔비디아·알파벳·메타·테슬라)만 따로 떼서 봐도 매출 성장률이 17%대로 전망된다. 당장은 기술주 주가가 주춤하고 있지만 그간 주가에 제동을 걸던 비관론이 많이 해소된 상황에서 3분기 호실적까지 발표되면 주가는 당연히 다시 오를 것이라고 본다.”
‌
엔비디아는 블랙웰 출시 지연으로 3분기 실적이 저조할 것 같다.
‌
“최근 엔비디아 실적 전망치가 알게 모르게 계속 오르고 있다. 3분기 가이던스가 매출 325억 달러(약 43조6800억 원), 주당순이익(EPS) 0.74달러였다. 이 자체로도 전년 동기 대비 80% 넘게 늘어나는 것인데, 월가에서 매주 EPS 전망치를 0.2%씩 올려 잡고 있다. 그 이유는 젠슨 황 엔비디아 최고경영자(CEO)가 국가별 독립 AI를 구축해야 한다는 ‘소버린AI’를 강조하고 AI 스타트업에 대한 지원 의사를 밝히는 등 AI 산업 저변을 넓히고 있기 때문이다. 블랙웰이 없으니 실적이 안 좋으리라고 예상하는 건 너무 시야가 좁은 판단이다. 이런 템포라면 EPS 0.8달러도 가능할 것 같다.”
‌
그럼에도 시장 눈에 차지 않으면 주가는 안 움직이지 않을까.
‌
“이 정도 실적도 ‘별것 아니다’라는 반응이면 할 말이 없다(웃음). 투자자들이 왜 답답해하는지는 안다. 오늘내일 당장 주가가 크게 움직이지 않는 게 투자자 입장에서는 답답한 것이다. 몇 배씩 오르던 주가가 멈칫거리는 데 대한 불만이다. 그런데 기업은 이미 알려진 사실로는 주가가 폭발적으로 오르기 어렵다. 그래서 이번 실적이 중요한데, 블랙웰 없이도 이 정도라는 것을 보여주면 시장이 ‘블랙웰이 나오면 더 좋겠구나’라는 관점을 가지면서 아마 탄력을 받을 것이다.”
‌
연말까지 160~180달러 주가 전망은 계속 유지하는 것인가.
‌
“그렇다. 현재 엔비디아 연간 EPS가 4달러다. 여기에 5년 평균 주가수익비율(PER)인 40을 곱하면 160달러다. 이 정도가 적정 주가 수준인 것이다. 그런데 지금 130달러 전후를 기록 중이니 훨씬 싸게 움직이고 있는 것이고, 이게 연말까지 정상화될 것이라고 본다(그래프 참조). 2030ㄷ턋()
년까지 더 장기적 관점에서 본다면 엔비디아 주가는 지금의 5배 이상으로 오를 것이다. 현 시장점유율이 유지된다는 가정 하에 엔비디아의 AI 관련 매출은 2030년 원화 베이스 2000조 원에 이를 것으로 추산된다. 엔비디아의 현 매출은 약 200조 원으로 10분의 1 수준이다. 그런데 2030년까지 엔비디아가 90% 시장점유율을 유지하는 건 무리가 있으니, 50%로 잡더라도 1000조 원이다. 그러면 200조 원 매출에서 1000조 원으로 5배가 커지는 것이다. 정말 간단한 계산만으로도 엔비디아 주가는 걱정할 게 없어 보인다. 6년간 5배 상승이면 결코 적지 않다. 다른 변수를 모두 제쳐놓고, 미국에서 주가는 결국 실적을 따라가기 때문에 숫자를 믿으면 된다. 지금 차트가 재미없는 것은 인정하지만 엔비디아만큼은 주가가 빠질 때마다 더 모아가는 전략이 맞다고 본다.”


“현금 비중 확대? 재진입 못 한다”
로보택시 공개를 앞둔 테슬라 주가 전망은 어떤가.
‌
“테슬라는 실력보다 기대를 많이 받는 종목이라서 현 주가 수준이 실적에 비해 비싼 편이다. 최근에는 말한 대로 로보택시 기대감이 너무 커져 있는 것 같다. 다만 한 가지 주목할 부분은 어제(현지 시간 10월 7일) 테슬라 주가가 빠졌다는 점이다. 로보택시 발표가 10일이고, 원래라면 주가가 떨어져선 안 된다. 그런데 이날 미국 사람들이 로보택시 이용을 꺼린다는 설문조사 결과가 나왔다. 실제로 미국 내 자율주행 사건·사고가 수백 건에 달하다 보니 ‘테슬라나 테슬라의 로보택시는 좋지만 나는 이용 안 할 것 같아’라는 갭이 생긴 것이다. 막연한 기대감에 주가가 오르다가 ‘아직 타겠다는 사람이 없네’라며 하락한 것으로 보인다. 앞으로도 주가에 이런 격차가 계속 작용하게 된다는 점을 염두에 둬야 한다. 또 테슬라는 현재 로보택시 외에 전기차, 로봇 등 모든 분야에서 상승을 이끌 만한 재료가 없는 상태다. 테슬라 주가 상승 원동력에 대해서는 퀘스천 마크가 있다.”
‌
현금 비중을 늘릴 때라는 전문가 조언이 많이 나온다. 이미 그것을 실천한 서학개미도 적잖다. 어떻게 보나.
‌
“내 경우에는 현금이 하나도 없다. 상당수 투자자가 무서우니까 주식을 팔았다가 저점에서 다시 진입하겠다는 생각을 하는데, 그게 사실상 불가능하다. 일례로 마이크론테크놀로지 주가가 이번 3분기 실적 발표 이후 15% 가까이 급등했다. 이렇게 전조 증상 없이 한 번에 주가를 끌어올리는 게 미국 장의 특징이라서 다시 들어갈 기회를 잡기가 쉽지 않다. 포트폴리오가 너무 한 섹터에만 쏠려 있다면 다른 가치주 등을 섞어서 분산하는 게 맞지 어디까지 빠질지, 언제 다시 오를지 아무도 모르는 상황에서 현금을 확보하라는 건 무책임한 얘기다. 또 지금 장이 안 가는 것 같아도 다 사상 최고치 바로 밑에서 움직이고 있다. 엔비디아도 한 번 오르기 시작하면 이틀 만에 전 고점 회복이 가능해 보인다. 너무 빠지면 빠진 대로 무서워서 못 사고, 갑자기 오르면 후회스러워서 못 살 테니 현금 비중을 늘리는 것은 추천하지 않는다.”   """
    
     
   
    main(articles, additional_keywords=['주식'])