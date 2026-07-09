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
    "Ask company knowledge...",
    disabled=st.session_state.is_processing
)

StreamlitUI.render_footer_note()

# ======================================
# USER MESSAGE
# ======================================

if (
    question
    and not st.session_state.is_processing
):

    ChatManager.add_message(
        role="user",
        content=question
    )

    # Disable the input immediately after submission.
    st.session_state.is_processing = True

    # Store the exact question currently being answered.
    st.session_state.pending_question = question

    st.rerun()

# ======================================
# ASSISTANT RESPONSE
# ======================================

messages = ChatManager.get_current_messages()

if (
    st.session_state.is_processing
    and st.session_state.pending_question
    and ChatManager.has_messages()
    and messages[-1]["role"] == "user"
):

    question = st.session_state.pending_question

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

        except Exception as e:

            loading.empty()

            error_message = (
                f"Error: {str(e)}"
            )

            ChatManager.add_message(
                role="assistant",
                content=error_message
            )

        finally:

            # Re-enable the input only after the assistant
            # response or error message has been saved.
            st.session_state.is_processing = False
            st.session_state.pending_question = None

            st.rerun()
