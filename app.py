import streamlit as st
import asyncio
from agents import create_shopping_agent
from connection import config
from agents import Runner

st.set_page_config(page_title="🛍️ Gemini Shopping Assistant", layout="centered")

st.title("🛍️ Gemini Shopping Assistant")
st.caption("Powered by Gemini + Agents SDK")

user_input = st.text_input("What would you like to buy?", placeholder="e.g. 2 red Nike shoes size 42")

if st.button("Submit"):
    if not user_input.strip():
        st.warning("Please enter something first.")
    else:
        agent = create_shopping_agent()

        with st.spinner("🧠 Understanding your request..."):
            result = asyncio.run(Runner.run(agent, input=user_input, run_config=config))

        st.success("✅ Extracted Shopping Info")
        st.json(result.output_text if hasattr(result, "output_text") else str(result))
