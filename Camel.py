import json
from json import JSONDecodeError
import logging

from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from camel.configs import ModelConfigFactory, BaseModelConfig
from camel.messages import (
    HumanMessage,
    SystemMessage,
    ChatMessage, AssistantChatMessage, UserChatMessage, AIMessage,
)

from tenacity import retry, stop_after_attempt, wait_exponential
from camel.messages import MessageType
from typing import List, Any
from camel.typing import ModelType, InteractionMode, RoleType
from FreeLLM import HuggingChatAPI, BardChatAPI  # FREE HUGGINGCHAT API
from FreeLLM import ChatGPTAPI  # FREE CHATGPT API
from FreeLLM import BingChatAPI  # FREE BINGCHAT API
import streamlit as st
from streamlit_chat_media import message

_logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="FREE AUTOGPT 🚀 by Intelligenza Artificiale Italia",
    page_icon="🚀",
    layout="wide",
    menu_items={
        "Get help": "https://www.intelligenzaartificialeitalia.net/",
        "Report a bug": "mailto:servizi@intelligenzaartificialeitalia.net",
        "About": "# *🚀  FREE AUTOGPT  🚀* ",
    },
)


st.markdown(
    "<style> iframe > div {    text-align: left;} </style>", unsafe_allow_html=True
)


def get_sys_msgs(_assistant_role_name: str, _user_role_name: str, _task: AssistantChatMessage):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(
        template=assistant_inception_prompt
    )
    _assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=_assistant_role_name,
        user_role_name=_user_role_name,
        task=_task,
    )[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(
        template=user_inception_prompt
    )
    _user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=_assistant_role_name,
        user_role_name=_user_role_name,
        task=_task,
    )[0]

    return AssistantChatMessage(role_name=_assistant_role_name, content=_assistant_sys_msg.content), \
        UserChatMessage(role_name=_user_role_name, content=_user_sys_msg.content)


class CAMELAgent:
    r"""Class for managing conversations of CAMEL Chat Agents.

    Args:
        system_message (SystemMessage): The system message for the chat agent.
        model (ModelType, optional): The LLM model to use for generating
            responses. (default :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any, optional): Configuration options for the LLM model.
            (default: :obj:`None`)
        message_window_size (int, optional): The maximum number of previous
            messages to include in the context window. If `None`, no windowing
            is performed. (default: :obj:`None`)
    """
    def __init__(
        self,
        system_message: ChatMessage,
        model_type: ModelType = ModelType.GPT_3_5_TURBO,
        model_config: BaseModelConfig = None,
    ) -> None:
        # self.system_message = system_message.content
        self.system_message = system_message
        self.role_name = system_message.role_name
        self.role_type = system_message.role_type
        self.meta_dict = system_message.meta_dict
        self.model_type = model_type
        self.model_config = model_config or ModelConfigFactory.get(model_type=model_type)
        self.chat_instance = self.get_chat_instance()
        self.terminated = False
        self.init_messages()

    def get_chat_instance(self, model_type: ModelType = None, model_config: BaseModelConfig = None):
        model_type = model_type or self.model_type
        model_config = model_config or self.model_config
        if model_type in (ModelType.GPT_3_5_TURBO, ModelType.GPT_4, ModelType.GPT_4_32k):
            if model_config.chat_session:
                chat_instance = ChatGPTAPI.ChatGPT(token=model_config.chat_token, conversation=model_config.chat_session,
                                                   model=model_config.sub_model)
            else:
                chat_instance = ChatGPTAPI.ChatGPT(token=model_config.chat_token, model=model_config.sub_model)
        elif model_type == ModelType.HUGGING_Chat:
            chat_instance = HuggingChatAPI.HuggingChat()
        elif model_type == ModelType.BING_CHAT:
            try:
                cookie_path = model_config.vaidate_cookie_path()
                chat_instance = BingChatAPI.BingChat(cookiepath=cookie_path, conversation_style="creative")
            except Exception as e:
                raise e
        elif model_type == ModelType.GOOGLE_BARD:
            chat_instance = BardChatAPI.BardChat(cookie=model_config.chat_token)
        else:
            raise ValueError(f"Invalid model type {model_type}.")
        return chat_instance

        # CG_TOKEN = os.getenv("CHATGPT_TOKEN", chat_token)
        # if CG_TOKEN is None:
        #     st.warning("The OpenAI token is missing.")
        #     st.stop()
        # if interaction_mode:
        #     if not chat_session:
        #         st.warning("You have to input an existing chat-id before starting chat.")
        #         st.stop()
        #     chat_instance = ChatGPTAPI.ChatGPT(token=CG_TOKEN, conversation=chat_session, model=sub_model)
        # else:
        #     chat_instance = ChatGPTAPI.ChatGPT(token=CG_TOKEN, model=sub_model)
        # return None

    def reset(self) -> List[MessageType]:
        r"""Resets the :obj:`ChatAgent` to its initial state and returns the
        stored messages.

        Returns:
            List[MessageType]: The stored messages.
        """
        self.terminated = False
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages: List[MessageType] = [self.system_message]

    def update_messages(self, message: SystemMessage) -> List[MessageType]:
        self.stored_messages.append(message)
        return self.stored_messages

    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(5))
    def step(
        self,
        input_message: SystemMessage,
    ) -> AssistantChatMessage:
        messages = self.update_messages(input_message)
        output_message = AssistantChatMessage(role_name=self.role_name, content=self.chat_instance(input_message.content))
        if output_message:
            self.update_messages(output_message)
            _logger.debug(f"AI Assistant:\n\n{output_message}\n\n")
        else:
            self.terminated = True
        return output_message

    def __repr__(self) -> str:
        r"""Returns a string representation of the :obj:`ChatAgent`.

        Returns:
            str: The string representation of the :obj:`ChatAgent`.
        """
        return f"ChatAgent({self.role_name}, {self.role_type}, {self.chat_instance})"


model_options = [v.value for v in ModelType.__members__.values()]
selected_model = st.selectbox("Select AI Model", options=model_options, key="model_selector")
col1, col2 = st.columns(2)
assistant_role_name = col1.text_input("Assistant Role Name", "Python Programmer")
user_role_name = col2.text_input("User Role Name", "Stock Trader")
task = st.text_area("Task", "Develop a trading bot for the stock market")
word_limit = st.number_input("Word Limit", 10, 1500, 500)

cg_col_1, cg_col_2, cg_col_3, cg_col_4 = st.columns(4)
is_openai_model = selected_model in (ModelType.GPT_3_5_TURBO.value, ModelType.GPT_4.value, ModelType.GPT_4_32k.value)
# sub_model = cg_col_1.selectbox("Sub_model", options=['GPT3.5-Turbo', 'GPT4'], index=0,
#                                disabled=not is_openai_model, key='sub_model_selector')
model_type = ModelType(selected_model)
model_config = ModelConfigFactory.get(model_type=model_type)
interaction_mode_options = [v.value for v in InteractionMode.__members__.values()]
interaction_mode = cg_col_2.selectbox("Interaction Mode", options=interaction_mode_options,
                                      index=interaction_mode_options.index(model_config.interaction_mode),
                                      key="interaction_mode", disabled=not is_openai_model)
if interaction_mode and interaction_mode != model_config.interaction_mode:
    model_config.interaction_mode = interaction_mode
chat_token = cg_col_3.text_input("Chat Token", key="chat_token", disabled=not is_openai_model or not model_config.model_type != ModelType.GOOGLE_BARD)
if chat_token and chat_token != model_config.chat_token:
    model_config.chat_token = chat_token
chat_session = cg_col_3.text_input("Chat Session", key="chat_session", disabled=interaction_mode != InteractionMode.CHAT)
if chat_session and chat_session != model_config.chat_session:
    model_config.chat_session = chat_session

upload_file = st.file_uploader("Choose the Bing Cookies JSON file", type="json",
                               disabled=model_config.model_type != ModelType.BING_CHAT)
if upload_file is not None:
    if model_config.update_cookie(upload_file):
        st.success("The Bing cookies file successfully uploaded.")
    else:
        st.error("Failed to upload the cookie file.")
        st.stop()

if st.button("Start working"):
    # if selected_model == model_options[0]:  # ChatGPT
    #     CG_TOKEN = os.getenv("CHATGPT_TOKEN", chat_token)
    #     if CG_TOKEN is None:
    #         st.warning("The OpenAI token is missing.")
    #         st.stop()
    #     if interaction_mode:
    #         if not chat_session:
    #             st.warning("You have to input an existing chat-id before starting chat.")
    #             st.stop()
    #         chat_instance = ChatGPTAPI.ChatGPT(token=CG_TOKEN, conversation=chat_session, model=sub_model)
    #     else:
    #         chat_instance = ChatGPTAPI.ChatGPT(token=CG_TOKEN, model=sub_model)

    # elif selected_model == model_options[1]:  # HuggingFace
    #     chat_instance = HuggingChatAPI.HuggingChat()

    # elif selected_model == model_options[2]:  # BingChat
    #
    #     if os.path.exists(bing_cookies_uploaded_path):
    #         cookie_path = bing_cookies_uploaded_path
    #     elif os.path.exists(bing_cookies_path):
    #         cookie_path = bing_cookies_path
    #     elif os.path.exists("cookiesBing.json"):
    #         cookie_path = "cookiesBing.json"
    #     else:
    #         st.warning(
    #             "File 'cookiesBing.json' not found! Create it and put your cookies in there in the JSON format."
    #         )
    #         st.stop()
    #     with open(cookie_path, "r") as file:
    #         try:
    #             file_json = json.loads(file.read())
    #         except JSONDecodeError:
    #             st.error(
    #                 "You did not put your cookies inside 'cookiesBing.json'! You can find the simple guide to get the "
    #                 "cookie file here: "
    #                 "https://github.com/acheong08/EdgeGPT/tree/master#getting-authentication-required."
    #             )
    #             st.stop()

    # elif selected_model == model_options[3]:  # Google Bard
    #     GB_TOKEN = os.getenv("BARDCHAT_TOKEN", gb_chat_token)
    #
    #     if GB_TOKEN and GB_TOKEN != "your-googlebard-token":
    #         os.environ["BARDCHAT_TOKEN"] = GB_TOKEN
    #     else:
    #         st.warning("GoogleBard Token EMPTY. Edit the .env file and put your GoogleBard token")
    #         st.stop()
    #     cookie_path = os.environ["BARDCHAT_TOKEN"]
    #     chat_instance = BardChatAPI.BardChat(cookie=cookie_path)
    #
    # else:
    #     st.warning(f"Unsupported model {selected_model}.")
    #     chat_instance = None

    task_specifier_sys_msg = SystemMessage(role_name=assistant_role_name, role_type=RoleType.ASSISTANT,
                                           content="You can make a task more specific.")
    task_specifier_prompt = """Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
    Please make it more specific. Be creative and imaginative.
    Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
    task_specifier_template = HumanMessagePromptTemplate.from_template(
        template=task_specifier_prompt
    )

    task_specify_agent = CAMELAgent(system_message=task_specifier_sys_msg, model_type=model_type,
                                    model_config=model_config)
    init_messages = task_specifier_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
        word_limit=word_limit,
    )
    if not init_messages:
        st.warning("No response from the AI service provider.")
        st.stop()
    task_specifier_msg = AssistantChatMessage(role_name=assistant_role_name, content=init_messages[0].content)
    specified_task_msg = task_specify_agent.step(task_specifier_msg)

    _logger.debug(f"Specified task: {specified_task_msg}")
    message(
        f"Specified task: {specified_task_msg}",
        allow_html=True,
        key="specified_task",
        avatar_style="adventurer",
    )

    specified_task = specified_task_msg

    assistant_inception_prompt = """Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles! Never instruct me!
    We share a common interest in collaborating to successfully complete a task.
    You must help me to complete the task.
    Here is the task: {task}. Never forget our task and to focus only to complete the task do not add anything else!
    I must instruct you based on your expertise and my needs to complete the task.

    I must give you one instruction at a time.
    It is important that when the . "{task}" is completed, you need to tell {user_role_name} that the task has completed and to stop!
    You must write a specific solution that appropriately completes the requested instruction.
    Do not add anything else other than your solution to my instruction.
    You are never supposed to ask me any questions you only answer questions.
    REMEMBER NEVER INSTRUCT ME! 
    Your solution must be declarative sentences and simple present tense.
    Unless I say the task is completed, you should always start with:

    Solution: <YOUR_SOLUTION>

    <YOUR_SOLUTION> should be specific and provide preferable implementations and examples for task-solving.
    Always end <YOUR_SOLUTION> with: Next request."""

    user_inception_prompt = """Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always instruct me.
    We share a common interest in collaborating to successfully complete a task.
    I must help you to complete the task.
    Here is the task: {task}. Never forget our task!
    You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:

    1. Instruct with a necessary input:
    Instruction: <YOUR_INSTRUCTION>
    Input: <YOUR_INPUT>

    2. Instruct without any input:
    Instruction: <YOUR_INSTRUCTION>
    Input: None

    The "Instruction" describes a task or question. The paired "Input" provides further context or information for the requested "Instruction".

    You must give me one instruction at a time.
    I must write a response that appropriately completes the requested instruction.
    I must decline your instruction honestly if I cannot perform the instruction due to physical, moral, legal reasons or my capability and explain the reasons.
    You should instruct me not ask me questions.
    Now you must start to instruct me using the two ways described above.
    Do not add anything else other than your instruction and the optional corresponding input!
    Keep giving me instructions and necessary inputs until you think the task is completed.
    It's Important wich when the task . "{task}" is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
    Never say <CAMEL_TASK_DONE> unless my responses have solved your task!
    It's Important wich when the task . "{task}" is completed, you must only reply with a single word <CAMEL_TASK_DONE>"""

    # define the role system messages
    assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)

    # AI ASSISTANT setup                           |-> add the agent LLM MODEL HERE <-|
    assistant_agent = CAMELAgent(system_message=assistant_sys_msg, model_type=model_type, model_config=model_config)

    # AI USER setup                      |-> add the agent LLM MODEL HERE <-|
    user_agent = CAMELAgent(system_message=user_sys_msg, model_type=model_type, model_config=model_config)

    # Reset agents
    assistant_agent.reset()
    user_agent.reset()

    # Initialize chats
    assistant_msg = HumanMessage(
        role_name=assistant_role_name,
        content=(
            f"{user_sys_msg}. "
            "Now start to give me introductions one by one. "
            "Only reply with Instruction and Input."
        )
    )

    user_msg = HumanMessage(role_name=user_role_name, content=f"{assistant_sys_msg.content}")
    user_msg = assistant_agent.step(user_msg)
    message(
        f"AI Assistant ({assistant_role_name}):\n\n{user_msg}\n\n",
        is_user=False,
        allow_html=True,
        key="0_assistant",
        avatar_style="pixel-art",
    )
    _logger.debug(f"Original task prompt:\n{task}\n")
    _logger.debug(f"Specified task prompt:\n{specified_task}\n")

    chat_turn_limit, n = 5, 0
    while n < chat_turn_limit:
        n += 1
        user_ai_msg = user_agent.step(assistant_msg)
        user_msg = HumanMessage(content=user_ai_msg.content, role_name=user_role_name)
        # print(f"AI User ({user_role_name}):\n\n{user_msg}\n\n")
        message(
            f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n",
            is_user=True,
            allow_html=True,
            key=str(n) + "_user",
        )

        assistant_ai_msg = assistant_agent.step(user_msg)
        assistant_msg = AIMessage(content=assistant_ai_msg.content, role_name=assistant_role_name)
        # print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg}\n\n")
        message(
            f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n",
            is_user=False,
            allow_html=True,
            key=str(n) + "_assistant",
            avatar_style="pixel-art",
        )
        if (
            "<CAMEL_TASK_DONE>" in user_msg.content
            or "task  completed" in user_msg.content
            or assistant_agent.terminated
            or user_agent.terminated
        ):
            message("Task completed!", allow_html=True, key="task_done")
            break
        if "Error" in user_msg.content:
            message("Task failed!", allow_html=True, key="task_failed")
            break
    if n >= chat_turn_limit:
        st.info("The chat turn limit is reached.")
