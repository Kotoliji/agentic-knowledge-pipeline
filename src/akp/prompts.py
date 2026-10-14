from langchain_core.prompts import ChatPromptTemplate

ROUTER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route questions for an assistant that answers from a company's internal "
            "documents (policies, processes, handbooks).\n"
            "Choose route='retrieve' for anything that could be answered from such documents, "
            "even if you are unsure the documents cover it.\n"
            "Choose route='direct' only for greetings, thanks, or questions about what the "
            "assistant can do.\n"
            "For 'retrieve', write search_query as a standalone, keyword-rich search query. "
            "For 'direct', leave search_query empty.",
        ),
        ("human", "{question}"),
    ]
)

DRAFT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the numbered passages below. Do not use outside "
            "knowledge.\n"
            "List in citations the numbers of the passages that state the facts you use.\n"
            "If the passages do not contain the answer, set answerable=false, leave citations "
            "empty and briefly say what information is missing.\n"
            "Be concise: answer in one to three sentences.\n\n"
            "Passages:\n{passages}",
        ),
        ("human", "{question}"),
    ]
)

CRITIC = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You check a drafted answer against the passages it was based on.\n"
            "Set supported=true only if the draft answers the question and every factual claim "
            "in it is stated in the cited passages. Otherwise set supported=false and describe "
            "the problem in issues.\n"
            "If the passages seem to lack the needed information, propose better_query: a "
            "different search query likely to find it. Leave better_query empty if the draft "
            "is supported or if another search would not help.\n\n"
            "Passages:\n{passages}",
        ),
        (
            "human",
            "Question: {question}\n\nDraft answer: {answer}\nCited passages: {citations}\n"
            "Draft marked answerable: {answerable}",
        ),
    ]
)

CHAT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for questions about a company's internal documents. "
            "Reply briefly and friendly. If the user asks what you can do, explain that you "
            "answer questions about company policies and processes, with citations.",
        ),
        ("human", "{question}"),
    ]
)
