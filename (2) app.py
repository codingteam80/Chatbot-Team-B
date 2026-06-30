import streamlit as st

from ui.streamlit_ui import StreamlitUI
from services.answer_service import AnswerService
from chat.chat_manager import ChatManager

# ======================================
# SETTINGS
# ======================================

DEBUG_MODE = False

# ======================================
# INITIALIZE
# ======================================

StreamlitUI.configure()

StreamlitUI.initialize_session()

ChatManager.initialize()

assistant = AnswerService()

# ======================================
# SIDEBAR
# ======================================

StreamlitUI.render_sidebar()

# ======================================
# MAIN PAGE
# ======================================

StreamlitUI.render_chat_history()

if ChatManager.is_current_chat_empty():

    StreamlitUI.show_welcome()

question = st.chat_input(
    "Ask company knowledge..."
)

StreamlitUI.render_footer_note()

# ======================================
# USER MESSAGE
# ======================================

if question:

    ChatManager.add_message(
        role="user",
        content=question
    )

    st.rerun()

# ======================================
# ASSISTANT RESPONSE
# ======================================

messages = ChatManager.get_current_messages()

if (
    ChatManager.has_messages()
    and messages[-1]["role"] == "user"
):

    question = messages[-1]["content"]

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching knowledge base..."
        ):

            try:

                response = assistant.ask(
                    question
                )

                answer = response.get(
                    "answer",
                    "No answer returned."
                )

                sources = response.get(
                    "sources",
                    []
                )

                chunks = response.get(
                    "chunks",
                    []
                )

                st.markdown(
                    answer
                )

                # ==================================
                # SOURCES
                # ==================================

                if sources:

                    with st.expander(
                        f"📄 Sources ({len(sources)})"
                    ):

                        for source in sources:

                            st.caption(
                                f"📄 {source}"
                            )

                # ==================================
                # ACTIONS
                # ==================================

                col1, col2 = st.columns(
                    2
                )

                with col1:

                    st.button(
                        "📋 Copy",
                        key=f"copy_{ChatManager.message_count()}"
                    )

                with col2:

                    st.download_button(
                        "📤 Export",
                        data=answer,
                        file_name="answer.txt",
                        mime="text/plain",
                        key=f"export_{ChatManager.message_count()}"
                    )

                # ==================================
                # DEBUG
                # ==================================

                if DEBUG_MODE:

                    with st.expander(
                        "Retrieved Chunks"
                    ):

                        for index, chunk in enumerate(
                            chunks,
                            start=1
                        ):

                            st.markdown(
                                f"### Chunk {index}"
                            )

                            st.write(
                                chunk.get(
                                    "metadata",
                                    {}
                                )
                            )

                            st.text_area(
                                label=f"chunk_{index}",
                                value=chunk.get(
                                    "text",
                                    ""
                                ),
                                height=180,
                                disabled=True
                            )

                ChatManager.add_message(
                    role="assistant",
                    content=answer,
                    sources=sources
                )

                st.rerun()

            except Exception as e:

                error_message = (
                    f"Error: {str(e)}"
                )

                ChatManager.add_message(
                    role="assistant",
                    content=error_message
                )

                st.rerun()