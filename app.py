import streamlit as st
from contextlib import nullcontext

from ui.streamlit_ui import StreamlitUI
from services.answer_service import AnswerService
from chat.chat_manager import ChatManager
from scripts.smart_build import (
    smart_build,
    check_changes
)
from config.settings import (
    TEST_EVIDENCE_MODE,
    OLLAMA_MODEL,
    EMBED_MODEL_NAME,
    RERANKER_MODEL,
    ENABLE_RERANKER,
    FINAL_TOP_K,
    MIN_RETRIEVAL_SCORE
)
from qa.evidence_logger import evidence_logger
from qa.test_case_registry import (
    evaluate_answer,
    get_next_run_number,
    get_test_case
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
# AUTOMATIC QA EVIDENCE
# ======================================
# No manual test-case input is required.
# Known QA questions are matched automatically.
# Unknown questions are still recorded as ad-hoc evidence.

if (
    TEST_EVIDENCE_MODE
    and "qa_auto_run_counts"
    not in st.session_state
):

    st.session_state.qa_auto_run_counts = {}

# ======================================
# MAIN PAGE
# ======================================

StreamlitUI.render_chat_history()

is_empty_chat = (
    ChatManager.is_current_chat_empty()
)

if is_empty_chat:

    StreamlitUI.show_welcome()

chat_input_container_key = (
    "chat_input_shell_empty"
    if is_empty_chat
    else "chat_input_shell_active"
)

# Dedicated fixed backdrop for active chats.
# It covers the full bottom area so messages do not
# remain visible behind the fixed input while scrolling.
if not is_empty_chat:

    st.markdown(
        '<div class="docubot-bottom-mask" aria-hidden="true"></div>',
        unsafe_allow_html=True
    )

# Clear the textbox only after a question has
# been accepted for processing.
if st.session_state.clear_chat_draft:

    st.session_state.chat_draft = ""
    st.session_state.clear_chat_draft = False

# The textbox remains editable while processing.
# Only the send action is disabled.
with st.container(
    key=chat_input_container_key
):

    with st.form(
        key="chat_input_form",
        clear_on_submit=False
    ):

        question = st.text_input(
            "Ask company knowledge...",
            key="chat_draft",
            label_visibility="collapsed",
            placeholder="Ask company knowledge..."
        )

        # Dedicated keyed wrapper makes the send button
        # reliable to position inside the textbox.
        with st.container(
            key="chat_send_button"
        ):

            submitted = st.form_submit_button(
                "➤",
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

regenerate_message_index = (
    st.session_state.regenerate_message_index
)

is_regeneration = (
    regenerate_message_index is not None
)

normal_request_ready = (
    ChatManager.has_messages()
    and messages[-1]["role"] == "user"
)

regeneration_ready = (
    is_regeneration
    and st.session_state.regenerate_chat_id
    == ChatManager.current_chat_id()
)

if (
    st.session_state.is_processing
    and st.session_state.pending_question
    and (
        normal_request_ready
        or regeneration_ready
    )
):

    question = st.session_state.pending_question

    regeneration_snapshot = None

    if is_regeneration:

        regeneration_snapshot = (
            ChatManager.begin_regeneration(
                regenerate_message_index,
                st.session_state.regenerate_chat_id
            )
        )

        if regeneration_snapshot is None:

            st.session_state.is_processing = False
            st.session_state.pending_question = None
            st.session_state.regenerate_message_index = None
            st.session_state.regenerate_chat_id = None

            st.rerun()

    response_container = (
        nullcontext()
        if is_regeneration
        else st.chat_message(
            "assistant",
            avatar="🤖"
        )
    )

    with response_container:

        loading = None

        if not is_regeneration:

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

        qa_run_started = False

        try:

            qa_test_case = None
            qa_cycle_number = 1

            if TEST_EVIDENCE_MODE:

                qa_test_case = get_test_case(
                    question
                )

                (
                    qa_run_number,
                    qa_total_runs,
                    qa_cycle_number
                ) = get_next_run_number(
                    question,
                    st.session_state.qa_auto_run_counts
                )

                evidence_logger.start_run(
                    test_case_id=qa_test_case[
                        "test_case_id"
                    ],
                    category=qa_test_case[
                        "category"
                    ],
                    description=qa_test_case[
                        "description"
                    ],
                    question=question,
                    expected_result=qa_test_case[
                        "expected_result"
                    ],
                    run_number=qa_run_number,
                    total_runs=qa_total_runs,
                    environment={
                        "Ollama Model":
                            OLLAMA_MODEL,

                        "Embedding Model":
                            EMBED_MODEL_NAME,

                        "Reranker Enabled":
                            ENABLE_RERANKER,

                        "Reranker Model":
                            (
                                RERANKER_MODEL
                                if ENABLE_RERANKER
                                else "Disabled"
                            ),

                        "Final Top-K":
                            FINAL_TOP_K,

                        "Minimum Retrieval Score":
                            MIN_RETRIEVAL_SCORE,

                        "Regeneration":
                            is_regeneration,

                        "Automatic QA Match":
                            qa_test_case.get(
                                "matched",
                                False
                            ),

                        "QA Cycle":
                            qa_cycle_number,
                    }
                )

                qa_run_started = True

            response = assistant.ask(
                question
            )

            if loading is not None:
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

            if (
                TEST_EVIDENCE_MODE
                and qa_run_started
            ):

                qa_status, qa_notes = (
                    evaluate_answer(
                        answer,
                        qa_test_case
                    )
                )

                evidence_logger.finish_run(
                    status=qa_status,
                    actual_result=answer,
                    notes=qa_notes
                )

                qa_run_started = False

            if not is_regeneration:

                st.markdown(
                    answer
                )

                if sources:

                    StreamlitUI.render_sources(
                        sources,
                        -1
                    )

            if DEBUG_MODE and not is_regeneration:

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

            if is_regeneration:

                ChatManager.complete_regeneration(
                    regeneration_snapshot,
                    answer,
                    sources
                )

            else:

                ChatManager.add_message(
                    role="assistant",
                    content=answer,
                    sources=sources
                )

        except Exception as e:

            if loading is not None:
                loading.empty()

            error_message = (
                f"Error: {str(e)}"
            )

            if (
                TEST_EVIDENCE_MODE
                and qa_run_started
            ):

                evidence_logger.finish_run(
                    status="ERROR",
                    actual_result=error_message,
                    error=str(e)
                )

                qa_run_started = False

            if is_regeneration:

                ChatManager.restore_regeneration(
                    regeneration_snapshot
                )

                print(error_message)

            else:

                ChatManager.add_message(
                    role="assistant",
                    content=error_message
                )

        finally:

            # Re-enable sending only after the answer
            # or error message has been saved.
            st.session_state.is_processing = False
            st.session_state.pending_question = None
            st.session_state.regenerate_message_index = None
            st.session_state.regenerate_chat_id = None

            st.rerun()
