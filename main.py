from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import load_tools

question = """
La stratégie sur l'infrastructure hébergeant ces applications spécifiques est également 
un élément majeur dans la réflexion de la DSI.
Quelles solutions d'hébergement s'offrent à la DSI ? Décrivez au moins deux solutions 
possibles en argumentant les avantages et inconvénients, en prenant en compte les 
contraintes et obligations potentielles liées à l'environnement hospitalier ?
"""

template = """Context: 
Du fait de besoins spécifiques non couverts par des applications proposées par les éditeurs 
spécialisés du marché, la direction du système d'information (DSI) d'un centre hospitalier
va devoir être en capacité de développer et déployer des applications spécifiques.
Pour cela, elle décide de s'organiser en faisant le choix de gérer en interne ces 
développements, en s'adossant aux équipes déjà en place, l'équipe Projets et l'équipe 
Infrastructure notamment.
Du fait des enjeux contextuels (réponse rapide aux demandes Métiers de l'hôpital), la DSI 
doit pouvoir mettre en service ces applications spécifiques de manière la plus agile possible.
Pour cela, la DSI décide de s'orienter vers une démarche DevOps.

Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="models/7B_ggml-model-q4_0.bin", callback_manager=callback_manager, verbose=True, n_ctx=8192
)

tool_names = ["requests", "wikipedia"]

llm_chain = LLMChain(prompt=prompt, llm=llm)

tools = load_tools(tool_names, llm=llm)

llm_chain.run(question)