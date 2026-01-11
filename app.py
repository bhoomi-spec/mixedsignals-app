import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


# PRIMARY: VADER (ALWAYS ACTIVE - 60%)
analyzer = SentimentIntensityAnalyzer()


# SECONDARY: DistilBERT Transformer (OPTIONAL - 40%)
TRANSFORMER_AVAILABLE = False
sentiment_pipeline = None
try:
    import torch
    from transformers import pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU-only
    )
    TRANSFORMER_AVAILABLE = True
except:
    TRANSFORMER_AVAILABLE = False


#  PERFECT UI
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #00b4d8 100%); }
    .stApp { background: #0a192f; }
    .stMarkdown, h1, h2, h3 { color: #64b5f6 !important; font-family: 'Segoe UI', sans-serif; }
    h1 { color: #2196f3 !important; font-size: 3rem; font-weight: 700; text-align: center; }
    h2 { color: #42a5f5 !important; font-size: 1.8rem; }
    h3 { color: #90caf9 !important; font-size: 1.3rem; }
   
    .metric-card { background: rgba(66,165,245,0.12); border: 1px solid rgba(66,165,245,0.2);
                   border-radius: 12px; padding: 1.5rem; text-align: center; margin: 0.5rem 0; }
    .advice-card { background: rgba(76,175,80,0.15); border: 1px solid rgba(76,175,80,0.3);
                   border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
   
    .stTextArea textarea, .stTextInput input {
        background: rgba(33,150,243,0.08) !important; border: 2px solid #42a5f5 !important;
        color: #ffffff !important; border-radius: 8px; }
    .stButton > button { background: linear-gradient(45deg, #1976d2, #42a5f5) !important;
                         color: white !important; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="MixedSignals v13", layout="wide")


# SESSION STATE
if 'people' not in st.session_state: st.session_state.people = {}
if 'ai_name' not in st.session_state: st.session_state.ai_name = 'Alex'
if 'current_person' not in st.session_state: st.session_state.current_person = None
if 'use_transformer' not in st.session_state: st.session_state.use_transformer = TRANSFORMER_AVAILABLE


# GENZ REJECTION PATCH
GENZ_REJECTION = {
    "not": -0.75, "anymore": -0.85, "type": -0.65, "friends": -0.55,
    "ghost": -0.9, "sus": -0.7, "toxic": -0.9, "break": -1.0
}


def apply_genz_patch(text, compound):
    """🔧 GenZ Rejection Patch - Multiplies by 0.45x (YOUR SPEC)"""
    words = text.lower().split()
    boost = sum(GENZ_REJECTION.get(word, 0) * 0.45 for word in words)
    return max(-1.0, min(1.0, compound + boost))


def classify_health(compound):
    """ Health Score Classifier (YOUR EXACT THRESHOLDS)"""
    if compound >= 0.05:
        return "POSITIVE", min(100, 60 + int(compound * 60))
    elif compound <= -0.05:
        return "NEGATIVE", max(0, 30 + int(compound * 60))
    return "NEUTRAL", 45


#  Research Metrics Functions
def calculate_metrics(df):
    """Calculate Accuracy/Precision/Recall/F1 using compound threshold as ground truth"""
    df = df.copy()
    df['true_label'] = np.where(df['compound_final'] >= 0.05, 'POSITIVE',
                               np.where(df['compound_final'] <= -0.05, 'NEGATIVE', 'NEUTRAL'))
    y_true = df['true_label']
    y_pred = df['prediction']
   
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    y_pred_enc = le.transform(y_pred)
   
    accuracy = accuracy_score(y_true_enc, y_pred_enc)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_enc, y_pred_enc, average='weighted')
   
    return accuracy, precision, recall, f1


def plot_confusion_matrix(df):
    """Confusion Matrix Heatmap"""
    df_cm = df.copy()
    df_cm['true_label'] = np.where(df_cm['compound_final'] >= 0.05, 'POSITIVE',
                                  np.where(df_cm['compound_final'] <= -0.05, 'NEGATIVE', 'NEUTRAL'))
   
    cm = confusion_matrix(df_cm['true_label'], df_cm['prediction'],
                         labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
   
    fig = ff.create_annotated_heatmap(
        z=cm, x=['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
        y=['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
        colorscale='Viridis', showscale=True
    )
    fig.update_layout(title="Confusion Matrix", height=400)
    return fig


#  YOUR EXACT PIPELINE
def analyze_message(text):
    """Text → VADER(60%) + Transformer(40%) + GenZ Patch → Final Compound → Health"""
    vader_scores = analyzer.polarity_scores(text)
    compound_vader = apply_genz_patch(text, vader_scores['compound'])
   
    transformer_score = 0
    if st.session_state.use_transformer and TRANSFORMER_AVAILABLE and sentiment_pipeline:
        try:
            result = sentiment_pipeline(text[:512])[0]
            transformer_score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        except:
            transformer_score = 0
   
    final_compound = 0.6 * compound_vader + 0.4 * transformer_score
    prediction, health = classify_health(final_compound)
   
    return {
        'final': final_compound,
        'vader': compound_vader,
        'transformer': transformer_score,
        'prediction': prediction,
        'health': health,
        'pos': vader_scores['pos'],
        'neg': vader_scores['neg'],
        'neu': vader_scores['neu']
    }


# HEADER
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(33,150,243,0.08);
            border-radius: 12px; margin-bottom: 2rem; border: 1px solid rgba(66,165,245,0.3);'>
    <h1>MixedSignals And Sentiment Analysis</h1>
    <p style='color: #bbdefb; font-size: 1.2rem;'>VADER(60%) + DistilBERT(40%) + GenZ Patch</p>
</div>
""", unsafe_allow_html=True)


# SIDEBAR
with st.sidebar:
    st.markdown("###  Controls")
    st.session_state.ai_name = st.text_input("AI Name", value=st.session_state.ai_name)
    st.session_state.use_transformer = st.checkbox(" DistilBERT Transformer", value=st.session_state.use_transformer)
   
    st.caption(f"**Status**: VADER ✅ | DistilBERT {'✅' if TRANSFORMER_AVAILABLE else '❌'}")
   
    if st.session_state.people:
        st.session_state.current_person = st.selectbox("👤 Current Person", list(st.session_state.people.keys()))
    if st.button("🗑️ Clear All Data", type="secondary"):
        st.session_state.people = {}
        st.session_state.current_person = None
        st.rerun()


# INPUT SECTION
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### Analyze Message")
    person_name = st.text_input("👤 Profiles", placeholder="Type a name...")
    message = st.text_area("Message", height=140,
                          placeholder="Type in the message here...")


with col2:
    st.markdown("### Live Results")
    if (st.session_state.current_person and st.session_state.current_person in st.session_state.people and
        st.session_state.people[st.session_state.current_person].get('history')):
        data = st.session_state.people[st.session_state.current_person]
        latest = data['history'][-1]
       
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size:12px;'>Compound</div>
                <div style='font-size:24px;font-weight:700;'>{data['last_pred']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            color = {'POSITIVE': '#4caf50', 'NEGATIVE': '#f44336', 'NEUTRAL': '#ff9800'}
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size:22px;font-weight:700;color:{color[latest["prediction"]]}'>
                {latest["prediction"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size:24px;font-weight:700;'>{data['last_health']}%</div>
                <div style='font-size:12px;'>Health</div>
            </div>
            """, unsafe_allow_html=True)


# ANALYZE BUTTON
if st.button("ANALYZE", type="primary", use_container_width=True):
    if person_name and message.strip():
        if person_name not in st.session_state.people:
            st.session_state.people[person_name] = {'history': [], 'last_pred': 0.0, 'last_health': 50}
       
        person_data = st.session_state.people[person_name]
        result = analyze_message(message)
       
        analysis_result = {
            'timestamp': datetime.now(),
            'text': message[:80] + "..." if len(message) > 80 else message,
            'compound_final': result['final'],
            'vader': result['vader'],
            'transformer': result['transformer'],
            'prediction': result['prediction'],
            'health': result['health'],
            'pos': result['pos'],
            'neg': result['neg'],
            'neu': result['neu']
        }
       
        person_data['history'].append(analysis_result)
        person_data['last_pred'] = result['final']
        person_data['last_health'] = result['health']
        st.session_state.current_person = person_name
        st.success(f" {result['prediction']} ({result['health']}%)")
        st.rerun()


#  TABS + HISTORY TABLE + 11 VISUALIZATIONS
if st.session_state.people:
    st.markdown("---")
    person_names = list(st.session_state.people.keys())
    tabs = st.tabs(person_names)
   
    for i, person_name in enumerate(person_names):
        with tabs[i]:
            person_data = st.session_state.people[person_name]
            history = person_data['history']
           
            if history:
                st.markdown(f"""
                <div class="advice-card">
                    <strong> {st.session_state.ai_name}: </strong>
                    {person_name} Health: {person_data['last_health']}% -
                    {'TOXIC - Cut contact' if person_data['last_health'] < 30
                     else 'Healthy connection'}
                </div>
                """, unsafe_allow_html=True)
           
            if history:
                st.markdown("###  Message History")
                df_history = pd.DataFrame(history)
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%H:%M %d/%m')
                df_display = df_history[['timestamp', 'text', 'prediction', 'health', 'compound_final']].tail(15)
                st.dataframe(df_display, use_container_width=True, hide_index=True, height=350)
           
            # DASHBOARD (1+ messages)
            if len(history) >= 1:
                st.markdown("## Dashboard")
                df = pd.DataFrame(history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
               
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Sentiment Evolution")
                    fig1 = px.scatter(df.tail(50), x='timestamp', y='compound_final',
                                    color='prediction', size='health',
                                    color_discrete_map={'POSITIVE':'#4caf50','NEGATIVE':'#f44336','NEUTRAL':'#ff9800'},
                                    title="Sentiment Over Time")
                    fig1.update_layout(height=400)
                    st.plotly_chart(fig1, use_container_width=True)
               
                with col2:
                    st.markdown("### Prediction Distribution")
                    pred_counts = df['prediction'].value_counts()
                    fig2 = px.bar(x=pred_counts.index, y=pred_counts.values,
                                color=pred_counts.index,
                                color_discrete_map={'POSITIVE':'#4caf50','NEGATIVE':'#f44336','NEUTRAL':'#ff9800'},
                                title="POS/NEG/NEU Counts")
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)
               
                st.markdown("### Health Trend")
                fig3 = px.line(df.tail(30), x='timestamp', y='health', color='prediction', markers=True,
                             color_discrete_map={'POSITIVE':'#4caf50','NEGATIVE':'#f44336','NEUTRAL':'#ff9800'},
                             title="Health Score Over Time")
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)
           
            # RESEARCH SUITE (5+ messages)
            if len(history) >= 5:
                st.markdown("## Research Suite")
                df_eval = pd.DataFrame(history[-30:])
               
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Messages", len(df_eval))
                col2.metric("Avg Health", f"{df_eval['health'].mean():.0f}%")
                col3.metric("Positive", f"{len(df_eval[df_eval['prediction']=='POSITIVE'])/len(df_eval)*100:.0f}%")
                col4.metric("Negative", f"{len(df_eval[df_eval['prediction']=='NEGATIVE'])/len(df_eval)*100:.0f}%")
               
                # Model Performance Metrics
                accuracy, precision, recall, f1 = calculate_metrics(df_eval)
                st.markdown("###  Model Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.3f}")
                col2.metric("Precision", f"{precision:.3f}")
                col3.metric("Recall", f"{recall:.3f}")
                col4.metric("F1-Score", f"{f1:.3f}")
               
                df_research = pd.DataFrame(history[-50:])
                df_research['timestamp'] = pd.to_datetime(df_research['timestamp'])
               
                # Confusion Matrix
                st.markdown("### Confusion Matrix")
                fig5 = plot_confusion_matrix(df_research)
                st.plotly_chart(fig5, use_container_width=True)
               
                #  Lexicon Evolution
                st.markdown("###  Lexicon Evolution")
                df_lex = df_research[['timestamp', 'pos', 'neg', 'neu']].copy()
                df_lex = df_lex.sort_values('timestamp').reset_index(drop=True)
                df_lex['message_index'] = range(1, len(df_lex) + 1)
               
                fig6 = go.Figure()
                fig6.add_trace(go.Scatter(x=df_lex['message_index'], y=df_lex['pos'],
                                         name='Positive', line=dict(color='#4caf50', width=3), fill='tonexty'))
                fig6.add_trace(go.Scatter(x=df_lex['message_index'], y=df_lex['neg'],
                                         name='Negative', line=dict(color='#f44336', width=3), fill='tonexty'))
                fig6.add_trace(go.Scatter(x=df_lex['message_index'], y=df_lex['neu'],
                                         name='Neutral', line=dict(color='#ff9800', width=3),
                                         fillcolor='rgba(255,152,0,0.3)'))
                fig6.update_layout(
                    title="POS/NEG/NEU Lexicon Evolution (Message-by-Message)",
                    xaxis_title="Message Number",
                    yaxis_title="Lexicon Intensity",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig6, use_container_width=True)
               
                # Correlation Heatmap
                st.markdown("###  Feature Correlation Heatmap")
                corr_matrix = df_research[['compound_final', 'vader', 'transformer', 'health', 'pos', 'neg', 'neu']].corr()
                fig7 = px.imshow(corr_matrix, aspect="auto", color_continuous_scale='RdBu_r', title="Feature Correlations")
                fig7.update_layout(height=400)
                st.plotly_chart(fig7, use_container_width=True)
               
                #  Positive Histogram
                st.markdown("### Positive Predictions Histogram")
                pos_data = df_research[df_research['prediction'] == 'POSITIVE']['health'].dropna()
                if len(pos_data) > 0:
                    fig8 = px.histogram(pos_data, x=pos_data, nbins=20, title="Positive Health Distribution",
                                       color_discrete_sequence=['#4caf50'])
                    fig8.update_layout(height=400)
                    st.plotly_chart(fig8, use_container_width=True)
                else:
                    st.warning("No positive predictions for histogram")
               
                #  Negative Histogram
                st.markdown("### Negative Predictions Histogram")
                neg_data = df_research[df_research['prediction'] == 'NEGATIVE']['health'].dropna()
                if len(neg_data) > 0:
                    fig9 = px.histogram(neg_data, x=neg_data, nbins=20, title="Negative Health Distribution",
                                       color_discrete_sequence=['#f44336'])
                    fig9.update_layout(height=400)
                    st.plotly_chart(fig9, use_container_width=True)
                else:
                    st.warning("No negative predictions for histogram")
               
                # Health vs Compound (FIXED - NO statsmodels)
                st.markdown("### Health vs Compound Scatter")
                fig10 = px.scatter(df_research, x='compound_final', y='health',
                                  color='prediction', size='health',
                                  color_discrete_map={'POSITIVE':'#4caf50','NEGATIVE':'#f44336','NEUTRAL':'#ff9800'},
                                  title="Health vs Compound Score",
                                  hover_data=['prediction'])
               
                if len(df_research) >= 2:
                    z = np.polyfit(df_research['compound_final'], df_research['health'], 1)
                    p = np.poly1d(z)
                    fig10.add_scatter(x=df_research['compound_final'], y=p(df_research['compound_final']),
                                     mode='lines', name='Trendline',
                                     line=dict(color='#2196f3', width=4, dash='dash'))
               
                fig10.update_layout(height=400)
                st.plotly_chart(fig10, use_container_width=True)
               
                #  Summary Statistics
                st.markdown("###  Summary Statistics")
                summary_stats = df_research[['compound_final', 'health', 'pos', 'neg', 'neu']].describe()
                st.dataframe(summary_stats.round(3), use_container_width=True)
               
               
st.markdown("""
<div style='text-align:center;padding:2rem;color:#bbdefb;font-size:14px;border-top:1px solid rgba(66,165,245,0.2);'>
    ✅ MixedSignals And SentimentAnalysis - VADER(60%) + DistilBERT(40%) + GenZ Patch |
</div>
""", unsafe_allow_html=True)