"""
Section 1: DR Grade Diagnosis Component
"""

import streamlit as st
import plotly.graph_objects as go
from frontend.config import DR_GRADE_NAMES, DR_GRADE_COLORS


def render_diagnosis_section(pred_grade: int, confidence: float, probs: list):
    """
    Render DR grade diagnosis with confidence and probability distribution.
    
    Args:
        pred_grade: Predicted grade (0-4)
        confidence: Prediction confidence (0-1)
        probs: Probability array for all 5 grades
    """
    st.subheader("📊 Section 1 — DR Grade Diagnosis")
    
    col_grade, col_conf = st.columns([1, 2])
    
    # ── Left: Grade Box ───────────────────────────────────────────────────────
    with col_grade:
        grade_name = DR_GRADE_NAMES[pred_grade]
        grade_color = DR_GRADE_COLORS[pred_grade]
        
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px;
                        border-radius:10px; border: 2px solid {grade_color};">
                <h2 style="color:{grade_color}; margin:0;">
                    {grade_name}
                </h2>
                <p style="font-size:18px; color:gray; margin:5px 0 0 0;">
                    Confidence: <b>{confidence*100:.1f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # ── Right: Probability Bar Chart ──────────────────────────────────────────
    with col_conf:
        fig_bar = go.Figure(go.Bar(
            x=probs * 100,
            y=DR_GRADE_NAMES,
            orientation="h",
            marker_color=DR_GRADE_COLORS,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
        ))
        
        fig_bar.update_layout(
            title="Grade Probability Distribution",
            xaxis_title="Probability (%)",
            xaxis=dict(range=[0, 115]),
            height=220,
            margin=dict(l=10, r=30, t=40, b=10),
            showlegend=False,
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)