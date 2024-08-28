from dotenv import load_dotenv
import os
import uuid
load_dotenv()
import sys
from authentication import *

import streamlit as st
from MonthlySales_Data import *

#from chain import chain as multimodal_rag_chain

def set_page_configs():
    from PIL import Image
    # Loading Image using PIL
    im = Image.open('../images/bot.png')
    # Adding Image to web app
    st.set_page_config(page_title="ActixOne ChatBot", page_icon = im)

    #st.set_page_config("Chat PDF")
    st.image("../images/title.PNG", width=400)
    st.header("Welcome to GenAI Powered ActixOne-Bot")
    st.image("../images/bot.PNG", width=100)  

def handle_message():
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = {}

    user_id = st.session_state.get("user_id", str(uuid.uuid4()))
    st.session_state["user_id"] = user_id

    if user_id not in st.session_state["messages"]:
        st.session_state["messages"][user_id] = [{"role": "assistant", "content": "Bring it on!"}]

    for msg in st.session_state["messages"][user_id]:
        st.chat_message(msg["role"], avatar="../images/bot.png").write(msg["content"])

    if prompt := st.chat_input():
        st.session_state["messages"][user_id].append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="../images/human2.png").write(prompt)

        message = st.session_state["messages"][user_id]

        user_question = next((item['content'] for item in reversed(message) if item['role'] == 'user'), None)
        print("user_quest:", user_question)

        #user_question = {"input_key": "Give me the sales data for the year 2021."}

        if user_question:
            answer = MonthlySales_agent_executor.invoke(user_question)
            print(answer)                                               
            st.session_state["messages"][user_id].append({"role": "user", "content": user_question})
            # msg = answer['output_text']
            msg = answer
            st.session_state["messages"][user_id].append({"role": "assistant", "content": msg})
            st.chat_message("assistant", avatar="../images/bot.png").write(msg)

def main():
    set_page_configs()

    status = check_password()

    if status:
        message = 'What kind of infor do you want from the sales deck?'
    handle_message()

if __name__ == "__main__":
    main()