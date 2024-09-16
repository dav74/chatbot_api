from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class Question(BaseModel):
    question: str

app = FastAPI()
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://pixees.fr/informatiquelycee/chatbot/",
    "https://pixees.fr/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
Settings.llm = Groq(temperature=0.8,model="llama-3.1-70b-versatile", api_key=os.getenv('GROQ_API_KEY'))
Settings.embed_model = HuggingFaceInferenceAPIEmbedding(model_name="OrdalieTech/Solon-embeddings-large-0.1", token = os.getenv('HF_API_KEY'))

try:
    storage_context = StorageContext.from_defaults(persist_dir = './storage')
    res_index = load_index_from_storage(storage_context)
    index_loaded = True
except:
    index_loaded = False

if not index_loaded:
    print('index creation, please wait...')
    res_doc = SimpleDirectoryReader('./documents').load_data()
    res_index = VectorStoreIndex.from_documents(res_doc)
    res_index.storage_context.persist(persist_dir = './storage')
    print('index creation completed')

chat_engine = res_index.as_chat_engine(
    chat_mode="context",
    system_prompt = (
        "Tu es un assistant spécialisé dans l'enseignement de la spécialité Numérique et sciences informatiques en classe de première et de terminal"
        'Tu as un bon niveau en langage Python'
        'Ton interlocuteur est un élève qui suit la spécialité nsi en première et en terminale'
        'Tu dois uniquement répondre aux questions qui concernent la spécialité numérique et sciences informatiques'
        "Tu ne dois pas faire d'erreur, répond à la question uniquement si tu es sûr de ta réponse"
        "si tu ne trouves pas la réponse à une question, tu réponds que tu ne connais pas la réponse et que l'élève doit s'adresser à son professeur pour obtenir cette réponse"
        "Tu dois uniquement aborder des notions qui sont aux programmes de la spécialité numérique et sciences informatiques (première et terminale), tu ne dois jamais aborder une notion qui n'est pas au programme"
        "si l'élève n'arrive pas à trouver la réponse à un exercice, tu ne dois pas lui donner tout de suite la réponse, mais seulement lui donner des indications pour lui permettre de trouver la réponse par lui même"
        "Tu peux lui donner la réponse à un exercice uniquement si l'élève te demande explicitement cette réponse"
        'Tu dois uniquement répondre en langue française'
        'Tu dois utiliser le format markdown pour tes réponses'
    )
)

@app.post('/ask')
def ask(question : Question):
    response = chat_engine.chat(question.question)
    return {"response":response.response}
