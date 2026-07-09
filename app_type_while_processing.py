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

# Clear the textbox only after a question has
# been accepted for processing.
if st.session_state.clear_chat_draft:

    st.session_state.chat_draft = ""
    st.session_state.clear_chat_draft = False

# The textbox remains editable while processing.
# Only the send action is disabled.
with st.form(
    key="chat_input_form",
    clear_on_submit=False
):

    input_column, send_column = st.columns(
        [12, 1],
        vertical_alignment="bottom"
    )

    with input_column:

        question = st.text_input(
            "Ask company knowledge...",
            key="chat_draft",
            label_visibility="collapsed",
            placeholder="Ask company knowledge..."
        )

    with send_column:

        submitted = st.form_submit_button(
            "➤",
            use_container_width=True,
            disabled=st.session_state.is_processing
        )

StreamlitUI.render_footer_note()

# ======================================
# USER MESSAGE
# ======================================

if submitted:

    clean_question = question.strip()

    # Server-side protection:
    # Enter or button clicks cannot submit another
    # question while the previous answer is processing.
    if (
        clean_question
        and not st.session_state.is_processing
    ):

        ChatManager.add_message(
            role="user",
            content=clean_question
        )

        st.session_state.is_processing = True
        st.session_state.pending_question = clean_question
        st.session_state.clear_chat_draft = True

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

            # Re-enable sending only after the answer
            # or error message has been saved.
            st.session_state.is_processing = False
            st.session_state.pending_question = None

            st.rerun()
