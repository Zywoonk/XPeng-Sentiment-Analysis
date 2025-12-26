import pandas as pd
import jieba
import matplotlib.pyplot as plt
import numpy as np
import os  # 

# Plotting Config
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- Configuration ---

# Plotting Config
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- Configuration (自动定位路径) ---
# 获取当前脚本(02.py)所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接出 csv 文件的完整路径
DATA_FILE = os.path.join(current_dir, 'data_sample.csv')

# Domain Dictionary (Aspect-Based Sentiment)
# 简单的关键词词典，用于定位评论是在聊哪个方面
ASPECTS = {
    'Intelligent Driving': ['NGP', '自动泊车', '辅助驾驶', '小P', '车机', '智能'],
    'Range & Battery': ['续航', '电耗', '充电', '掉电', '电池'],
    'Space & Comfort': ['空间', '后排', '座椅', '舒适', '隔音', '胎噪', '味道', '底盘'],
    'Design & Interior': ['颜值', '外观', '内饰', '设计', '车漆', '掀背'],
    'Service & Cost': ['销售', '服务', '交付', '性价比', '价格', '配置']
}

# Simple Sentiment Words (Simplification for demo)
POSITIVE_WORDS = ['大', '帅', '强', '好', '扎实', '不错', '快', '精准', '友好', '流畅', '无敌', '美']
NEGATIVE_WORDS = ['塑料', '一般', '大', '硬', '颠', '薄', '吵']
# Note: In a real project, we would use a pre-trained model (e.g., SnowNLP or BERT)
# Here we use a rule-based approach for transparency and speed.

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} reviews from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def analyze_sentiment(df):
    """
    Calculate sentiment score for each aspect based on keyword matching.
    """
    scores = {k: 0 for k in ASPECTS.keys()}
    counts = {k: 0 for k in ASPECTS.keys()}
    
    print("Analyzing sentiment...")
    
    for comment in df['content']:
        # Segmentation
        words = list(jieba.cut(comment))
        
        # Check which aspect is mentioned
        for aspect, keywords in ASPECTS.items():
            for kw in keywords:
                if kw in comment:
                    # Found a mention, now check sentiment
                    score = 0
                    for w in words:
                        if w in POSITIVE_WORDS: score += 1
                        if w in NEGATIVE_WORDS: score -= 1 # Simple logic
                    
                    # If score is 0 (neutral), we give a default positive bias for this demo
                    # assuming people talk about features they like mostly
                    final_score = score if score != 0 else 0.5
                    
                    scores[aspect] += final_score
                    counts[aspect] += 1
                    break # Count aspect only once per comment to avoid double counting

    # Calculate Average
    results = {}
    for k in scores:
        results[k] = scores[k] / counts[k] if counts[k] > 0 else 0
        
    return results

def plot_radar_chart(results):
    """
    Generate a Radar Chart (Spider Chart) for business insight.
    """
    labels = list(results.keys())
    values = list(results.values())
    
    # Close the loop for radar chart
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw logic
    ax.plot(angles, values, color='#1890ff', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='#1890ff', alpha=0.25)
    
    # Fix axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    # Range
    ax.set_ylim(-1, 2)
    
    plt.title('XPeng User Sentiment Analysis (Aspect-Level)', size=15, color='#333', y=1.05)
    
    plt.tight_layout()
    plt.savefig('sentiment_radar.png')
    print("✅ Chart saved: sentiment_radar.png")
    plt.show()

if __name__ == "__main__":
    df = load_data(DATA_FILE)
    if not df.empty:
        results = analyze_sentiment(df)
        print("Analysis Result:", results)
        plot_radar_chart(results)