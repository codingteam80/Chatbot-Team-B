import streamlit as st

from ui.streamlit_ui import StreamlitUI
from services.answer_service import AnswerService
from chat.chat_manager import ChatManager
from scripts.smart_build import (
    smart_build,
    check_changes
)

# ======================================
# SETTINGS
# ======================================

DEBUG_MODE = False

# ======================================
# INITIALIZE
# ======================================

StreamlitUI.configure()

StreamlitUI.initialize_session()

# ======================================
# KNOWLEDGE BASE STATUS
# ======================================

if "kb_checked" not in st.session_state:

    st.session_state.kb_outdated = (
        check_changes()
    )

    st.session_state.kb_checked = True

if st.session_state.kb_outdated:

    st.warning(
        "Knowledge base has changed. Rebuild is required."
    )

    if st.button(
        "Rebuild Knowledge Base"
    ):

        smart_build()

        st.session_state.kb_outdated = False

        st.success(
            "Knowledge base updated."
        )

        st.rerun()

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

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        loading = st.empty()

        loading.markdown(
            """
            <div class="loading-bubble">
                <div class="loading-spinner"></div>
                <div>Searching knowledge base...</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            response = assistant.ask(
                question
            )

            loading.empty()

            answer = response.get(
                "answer",
                "No answer returned."
            )

            sources = response.get(
                "sources",
                []
            )

            print("\n===== SOURCES =====")
            print(sources)
            print("===================\n")

            chunks = response.get(
                "chunks",
                []
            )

            st.markdown(
                answer
            )

            if sources:

                StreamlitUI.render_sources(
                    sources,
                    -1
                )

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

            loading.empty()

            error_message = (
                f"Error: {str(e)}"
            )

            ChatManager.add_message(
                role="assistant",
                content=error_message
            )

            st.rerun()