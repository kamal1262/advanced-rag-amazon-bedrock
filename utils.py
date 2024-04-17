import os

import logging
import sys
from pprint import pprint
import openai
import streamlit as st
from llama_index.llms.bedrock import Bedrock

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    Document
)



# from llama_index.llms import OpenAI, Anthropic
# from openai import OpenAI

from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.text_splitter import SentenceSplitter
# from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.schema import MetadataMode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# using openai 
""" 
openai_key = ""
os.environ["OPENAI_API_KEY"] = openai_key
openai.api_key = os.getenv("OPENAI_API_KEY")
"""
# use amazon bedrock , you need to pass your AWS profile here
llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=os.getenv("profile_name"))

# llm = Bedrock(
#     model="amazon.titan-text-express-v1",
#     aws_access_key_id="AWS Access Key ID to use",
#     aws_secret_access_key="AWS Secret Access Key to use",
#     aws_session_token="AWS Session Token to use",
#     aws_region_name="AWS Region to use, eg. us-east-1",
# )


#############################################################################################
#            vector index files
#############################################################################################
# generate the indexes using the notebook

index_path = '/index_with_rottnest'
calendar_index_path = '/index_calendar'

##############################################################################################
# SC_retrieved_sentence = StorageContext.from_defaults(persist_dir=index_path)
# retrieved_sentence_index = load_index_from_storage(SC_retrieved_sentence)

##############################################
#            query engine tools               
##############################################

def load_index(): 
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.selectors import PydanticSingleSelector
    from llama_index.core.tools import QueryEngineTool

    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        load_index_from_storage,
        StorageContext,
        ServiceContext,
        Document
    )


    # rebuild storage context
    calendar_retrieved_sentence = StorageContext.from_defaults(persist_dir= calendar_index_path)
    book_retrieved_sentence = StorageContext.from_defaults(persist_dir= index_path)

    # load index
    calendar_retrieved_index = load_index_from_storage(calendar_retrieved_sentence)
    book_retrieved_index = load_index_from_storage(book_retrieved_sentence)
    calendar_query_engine = calendar_retrieved_index.as_query_engine(        
        similarity_top_k=7,
        verbose=True,
        # the target key defaults to `window` to match the node_parser's default
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )
    book_query_engine = book_retrieved_index.as_query_engine(
        similarity_top_k=7,
        verbose=True,
        # the target key defaults to `window` to match the node_parser's default
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    calendar_tool = QueryEngineTool.from_defaults(
        query_engine=calendar_query_engine,

        description=(
            "meeting details for next week"
            "meeting pre-reads, important schedules, planning"
        ),
    )

    book_tool = QueryEngineTool.from_defaults(
        query_engine=book_query_engine,
        description=(
            "Rottnest island, personal diary, book"
            " China history and tea trade"
        ),
    )

    query_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(),
        query_engine_tools=[
            calendar_tool,
            book_tool,
        ],
    )

    return query_engine
##############################################
#            query engine tools               
##############################################

# Check if 'data_loaded' is in session state
if 'data_loaded' not in st.session_state:
    # Load data only if it hasn't been loaded before
    query_engine = load_index()
    st.session_state.data_loaded = True
else:
    # Use previously loaded data
    print(f"index already loaded {query_engine}")

def query_local_index(question):

    # question_with_prompt = s_prompt.format(question=question)

    response = query_engine.query(
        question
        )
    print(str(response))
    return response


def snowflake_answer(question, retrieved_sentence_index):
    

    sentence_query_engine = retrieved_sentence_index.as_query_engine(
        similarity_top_k=5,
        verbose=True,
        # the target key defaults to `window` to match the node_parser's default
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    # question = "Something happened in the United States 10 years after the first American ships sailed for China which could have made it more expensive to purchase tea. what happened that year? Try to break down your answer into steps."

    sentence_response = sentence_query_engine.query(
        question
    )

    return sentence_response


# question = "when did I visited Rottnest Island?"

# sentence_response = sentence_query_engine.query(
#     question
# )
# print(sentence_response)
