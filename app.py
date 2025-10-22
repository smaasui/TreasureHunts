import streamlit as st
import json
import asyncio
from agents import Tool, RunResult  # assuming your custom agents setup
from backend import ShoppingAgent  # import your agent class

st.set_page_config(page_title="Voice to JSON Agent", page_icon="🛒", layout="centered")

st.title("🛍️ Voice to JSON Shopping Agent")
st.write("Speak or type your order naturally (e.g., _'menu ek joota neelay rang da aur ek kilo doodh'_).")

# Input
user_input = st.text_area("🗣️ Enter your order", placeholder="menu ek joota neelay rang da aur ek kilo doodh")

# Run Agent
if st.button("🔍 Extract Items"):
    if not user_input.strip():
        st.warning("Please enter an order text first.")
    else:
        with st.spinner("Processing your request..."):
            try:
                # Run your agent asynchronously
                async def run_agent():
                    agent = ShoppingAgent(name="Shopping Agent")
                    result: RunResult = await agent.run(user_input)
                    return result

                # Execute event loop safely
                result = asyncio.run(run_agent())

                # Display structured JSON
                st.subheader("🧾 Extracted JSON Output")
                st.json(result.final_output if isinstance(result.final_output, dict) else result.final_output)

                # Optionally show intermediate details
                with st.expander("🔍 Agent Details"):
                    st.write(result)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
