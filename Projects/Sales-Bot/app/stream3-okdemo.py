from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def clear_submit():
    """
    Clear the Submit Button State
    """
    st.session_state["submit"] = False

def is_valid_code(code: str) -> bool:
    """
    Validate the given code string for correct Python syntax.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return False

st.set_page_config(page_title="LangChain: Chat with pandas DataFrame", page_icon="🦜")
st.title("🦜 LangChain: Chat with pandas DataFrame")

df = pd.read_csv('MonthlySales_Data.csv')

openai_api_key = os.environ.get('OPENAI_API_KEY')

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "system", "content": "whenever the question is about plot, chart, visualization, visual, \
                                      in addition to the data always give python code to generate the required plots\
                                      In your python code always define the dataframe df. Never assume that it already is defined.\
                                      Your code must NEVER ever say 'Assuming df is already defined'\
                                      when asked to plot multiple graphs or plots. plot them all one by one. \
                                      Never miss them. in each code define the dataframe as df = pd.DataFrame(data) first"})
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(
        temperature=0, model="gpt-4o-mini", openai_api_key=openai_api_key, streaming=True
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        max_iterations=40
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
        
        # Remove backticks from the code if present
        # cleaned_response = response.strip("```").strip()
        # Check if the response contains plot code
        if "plt.show()" in response or "sns." in response:
            print("i need to plottttttt"+response)
            print("hurrrrrrrrey i have plot code ")
            import re
            pattern = r"```python(.*?)```"
            matches = re.findall(pattern, response, re.DOTALL)
            code_block = [match.strip() for match in matches]

            print("here is the cleannnnnnnnnnnnnnn code:\n")
            print(code_block)
            for code in code_block:
                if is_valid_code(code):
                    exec_globals = {'pd': pd, 'plt': plt, 'sns': sns}
                    print(code)
                    exec(code, exec_globals)
                    st.pyplot(plt)
                    plt.clf()
                else:
                    print("Invalid code detected. Skipping execution.")
            
            print("Plotting attempted!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
