import json
import re
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import platform
import matplotlib.font_manager as fm
import matplotlib as mpl
import fasttext
from itertools import combinations
from openai import OpenAI


OPENAI_API_KEY = ''

nltk.download('punkt')

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

def split_text(text, max_length=4000):
    sentences = text.split('. ')
    current_chunk = []
    current_length = 0
    chunks = []

    for sentence in sentences:
        if current_length + len(sentence) + 1 > max_length:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence) + 1

    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

def get_ai_keywords(system_message, user_message):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)  # Instantiate the OpenAI client
        response = client.chat.completions.create(
            model='gpt-4o-mini-2024-07-18',
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': user_message}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "keyword_extraction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["keywords"],
                        "additionalProperties": False
                    }
                }
            }
        )

        response_json = json.loads(response.choices[0].message.content)
        return response_json.get("keywords", [])
    except Exception as e:
        print(f"Error with AI model: {e}")
        return []

def get_keywords_from_ai(article: str, system_message: str) -> list:
    # chunks = split_text(article)
    all_keywords = []

    # for chunk in chunks:
    keywords = get_ai_keywords(
        system_message=system_message,
        user_message=f"Extract important keywords from the following text. Respond in JSON format with a 'keywords' field: {article}"
        )
    all_keywords.extend(keywords)

    unique_keywords = list(set(all_keywords))
    return unique_keywords

def build_relationship_graph(model, keywords, additional_keywords=[]):
    G = nx.Graph()
    all_keywords = set(keywords) | set(additional_keywords)
    G.add_nodes_from(all_keywords)

    for combination in combinations(all_keywords, 2):
        vector1 = model.get_word_vector(combination[0])
        vector2 = model.get_word_vector(combination[1])
        if vector1 is not None and vector2 is not None:
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
    edge_widths = [weight * 2 for weight in weights]

    plt.figure(figsize=(8, 8))
    nx.draw_networkx(G, pos, with_labels=True, node_size=1500, node_color='skyblue', 
                     font_size=10, font_weight='bold', width=edge_widths, edge_color=edge_colors, 
                     edge_cmap=plt.cm.plasma, alpha=0.8, font_family=font_prop.get_name())
    plt.title("Keyword Relationship Graph (FastText)", fontsize=16)
    plt.show()

def main(article, additional_keywords=None):
    set_font()
    system_message = "You are an assistant that extracts keywords from text and provides responses in JSON format."
    keywords = get_keywords_from_ai(article, system_message)

    if not keywords:
        print("No keywords found.")
        return

    model = fasttext.load_model("model/facebook_ko.bin")
    G = build_relationship_graph(model, keywords, additional_keywords)
    visualize_graph(G)

if __name__ == "__main__":
    article = """기술주를 중심으로 미국 뉴욕증시가 지지부진한 흐름을 나타내고 있다. ‘인공지능(AI) 거품론’ ‘반도체 고점론’ 등 비관론이 퍼지면서 전체 증시가 한 차례 출렁인 가운데 11월 대선, 연방준비제도(연준)의 금리인하 속도 등 불확실한 요인들이 하방 압력을 가하고 있기 때문이다. 장우석 유에스스탁 부사장은 10월 8일 인터뷰에서 “이럴 때 엔비디아를 사 모아야 한다”고 말했다. 최근 서학개미는 기존에 선호하던 기술주를 대거 처분하고 배당주, 현금 등 안전자산 쪽으로 투자 노선을 갈아탄 상태인데, 이에 대해 “지금은 떠날 타이밍이 아니다”라고 조언한 것이다."""
    main(article, additional_keywords=['주가'])
