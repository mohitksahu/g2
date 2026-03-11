"""
Section 2: 12-Month Progression Risk Component
"""

import streamlit as st
from frontend.config import RISK_THRESHOLDS


def render_progression_section(prog_risk: float):
    """
    Render progression risk assessment with clinical recommendations.
    
    Args:
        prog_risk: Progression probability (0-1)
    """
    st.subheader("📈 Section 2 — 12-Month Progression Risk")
    
    # ── Determine Risk Category ───────────────────────────────────────────────
    if prog_risk < RISK_THRESHOLDS["low"]:
        risk_label = "LOW RISK"
        risk_color = "#2ecc71"
        recommendation = "Routine follow-up in 12 months."
        risk_emoji = "✅"
    elif prog_risk < RISK_THRESHOLDS["moderate"]:
        risk_label = "MODERATE RISK"
        risk_color = "#f39c12"
        recommendation = "Follow-up in 6 months, optimize glycemic control."
        risk_emoji = "⚠️"
    elif prog_risk < RISK_THRESHOLDS["high"]:
        risk_label = "HIGH RISK"
        risk_color = "#e74c3c"
        recommendation = "Refer to ophthalmologist within 4 weeks."
        risk_emoji = "🔴"
    else:
        risk_label = "VERY HIGH RISK"
        risk_color = "#7b241c"
        recommendation = "Urgent ophthalmology referral within 1 week."
        risk_emoji = "🚨"
    
    # ── Display Risk and Recommendation ───────────────────────────────────────
    col_risk, col_rec = st.columns(2)
    
    with col_risk:
        st.markdown(
            f"""
            <div style="text-align:center; padding:30px;
                        border-radius:10px; background:{risk_color}22;
                        border: 2px solid {risk_color};">
                <h1 style="color:{risk_color}; margin:0; font-size:56px;">
                    {prog_risk*100:.1f}%
                </h1>
                <h3 style="color:{risk_color}; margin:8px 0 0 0;">
                    {risk_emoji} {risk_label}
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col_rec:
        st.markdown(
            f"""
            <div style="padding:30px; border-radius:10px;
                        background:#f8f9fa; height:100%; 
                        border-left:4px solid {risk_color};">
                <h4 style="margin-top:0;">Clinical Recommendation</h4>
                <p style="font-size:18px;">{recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )