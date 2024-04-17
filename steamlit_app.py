from llama_index.llms.openai import OpenAI
import streamlit as st
from utils import snowflake_answer, calendar_index_path, index_path, query_local_index


st.title("Chatting with Your Private Data: Advanced RAG")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )

        message_placeholder = st.empty()
        # putting a spinning icon to show that the query is in progress
        with st.status("Determining the best possible answer!", expanded=True) as status:
            # passing the question into the snowflake_answer function, which later invokes the llm
            # answer = snowflake_answer(prompt, retrieved_sentence_index)
            answer = query_local_index(prompt)
            # writing the answer to the front end
            message_placeholder.markdown(f""" Answer:
                            {answer}""")

        # response = st.write_stream(stream)
        status.update(label="Question Answered...", state="complete", expanded=False)
    st.session_state.messages.append({"role": "assistant", "content": answer})
