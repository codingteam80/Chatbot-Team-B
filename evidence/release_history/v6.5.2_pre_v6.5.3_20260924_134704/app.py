import streamlit as st
from contextlib import nullcontext

from ui.streamlit_ui import StreamlitUI
from services.answer_service import AnswerService
from chat.chat_manager import ChatManager
from scripts.smart_build import (
    smart_build,
    get_update_plan,
)
from config.settings import (
    TEST_EVIDENCE_MODE,
    OLLAMA_MODEL,
    OLLAMA_FAST_MODEL,
    OLLAMA_COMPLEX_MODEL,
    OLLAMA_FORCED_MODEL,
    HYBRID_LLM_ROUTING_ENABLED,
    EMBED_MODEL_NAME,
    RERANKER_MODEL,
    ENABLE_RERANKER,
    FINAL_TOP_K,
    MIN_RETRIEVAL_SCORE,
    LAN_SERVER_MODE,
    ALLOW_WEB_KB_UPDATE
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


def _friendly_request_error(error):
    """Return a concise user-facing message while keeping raw detail in logs."""

    name = type(error).__name__.casefold()
    detail = str(error or "").casefold()
    connection_markers = (
        "connecterror", "connectionerror", "connection refused",
        "connection reset", "server disconnected", "remoteprotocolerror",
        "failed to establish", "ollama",
    )
    timeout_markers = ("timeout", "timed out", "readtimeout", "connecttimeout")

    if any(marker in name or marker in detail for marker in timeout_markers):
        return (
            "The local AI service took too long to respond. Please retry once. "
            "If it continues, restart Ollama and DocuBot."
        )

    if any(marker in name or marker in detail for marker in connection_markers):
        return (
            "The local AI service connection was interrupted. Please retry once. "
            "If it continues, restart Ollama and DocuBot."
        )

    return (
        "DocuBot could not complete this request. Please retry. "
        "If the issue continues, check the local DocuBot/Ollama runtime."
    )


# ======================================
# INITIALIZE
# ======================================

StreamlitUI.configure()

StreamlitUI.initialize_session()

# ======================================
# KNOWLEDGE BASE STATUS
# ======================================

if not st.session_state.get("kb_checked", False):
    kb_plan = get_update_plan()
    st.session_state.kb_outdated = kb_plan["mode"] != "noop"
    st.session_state.kb_update_mode = kb_plan["mode"]
    st.session_state.kb_update_reason = kb_plan.get("reason")
    st.session_state.kb_checked = True

if st.session_state.kb_outdated:
    full_rebuild_required = (
        st.session_state.get("kb_update_mode") == "full_rebuild"
    )

    # In centralized LAN mode, employee browser sessions must never race a
    # knowledge-base write. Server maintenance is performed locally by the
    # administrator with Update_DocuBot_Knowledge_Base.bat while DocuBot is
    # stopped. Normal single-PC mode preserves the certified web update flow.
    web_kb_update_allowed = (
        not LAN_SERVER_MODE
        or ALLOW_WEB_KB_UPDATE
    )

    if not web_kb_update_allowed:
        reason = st.session_state.get("kb_update_reason")
        message = (
            "Knowledge base changes detected. The central DocuBot server is "
            "in read-only employee mode. Ask the server administrator to stop "
            "DocuBot, run Update_DocuBot_Knowledge_Base.bat, then restart the server."
        )
        if full_rebuild_required and reason:
            message += f" Maintenance detail: {reason}"
        st.warning(message)

    else:
        if full_rebuild_required:
            st.warning(
                "Knowledge base maintenance is required before new source changes can be used safely. "
                + (st.session_state.get("kb_update_reason") or "A full rebuild is required.")
            )
            rebuild_confirmed = st.checkbox(
                "I understand that Update Knowledge Base will safely rebuild all source documents this time.",
                key="kb_smart_full_rebuild_confirm",
            )
        else:
            st.warning(
                "Knowledge base changes detected. The finalized Qdrant production profile will perform a safe transactional rebuild before publishing the updated index."
            )
            rebuild_confirmed = True

        if st.button(
            "Update Knowledge Base",
            key="kb_smart_update",
            disabled=not rebuild_confirmed,
        ):
            with st.spinner("Updating technical knowledge base safely..."):
                update_ok = smart_build()

            if update_ok:
                st.session_state.kb_checked = False
                st.success("Knowledge base updated.")
                st.rerun()
            else:
                st.error(
                    "Knowledge base update could not complete. The previous working "
                    "index is still active. If this repeats, run "
                    "Setup_DocuBot_Production_Environment.bat and retry. "
                    "Technical details are saved under logs\\kb_update."
                )

# Start quality-neutral runtime prewarming only when the current KB is healthy.
# The worker returns immediately; exact structured-only answers remain available
# even while semantic resources warm in the background.
if not st.session_state.kb_outdated:
    from runtime.prewarm import start_background_prewarm
    start_background_prewarm()

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

        question = st.text_area(
            "Ask company knowledge...",
            key="chat_draft",
            label_visibility="collapsed",
            placeholder="Ask company knowledge...",
            # Streamlit enforces a 68px minimum for st.text_area on the
            # production Windows version. CSS/JS still owns the compact
            # visible composer height after the widget is created.
            height=68,
        )

        # Dedicated keyed wrapper makes the send button
        # reliable to position inside the textbox.
        with st.container(
            key="chat_send_button"
        ):

            submitted = st.form_submit_button(
                "↑",
                disabled=st.session_state.is_processing
            )

# Browser keyboard contract: Enter sends; Shift+Enter inserts a newline.
# The textarea remains editable while processing, while plain Enter cannot
# submit because the send button is disabled during that state.
StreamlitUI.install_chat_input_keyboard_behavior()

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
                            (
                                OLLAMA_FORCED_MODEL
                                or (
                                    f"Hybrid: {OLLAMA_FAST_MODEL} / "
                                    f"{OLLAMA_COMPLEX_MODEL}"
                                )
                            ),

                        "LLM Routing":
                            (
                                "Forced model override"
                                if OLLAMA_FORCED_MODEL
                                else (
                                    "Hybrid"
                                    if HYBRID_LLM_ROUTING_ENABLED
                                    else "Fast model only"
                                )
                            ),

                        "Fast LLM Model":
                            OLLAMA_FAST_MODEL,

                        "Complex LLM Model":
                            OLLAMA_COMPLEX_MODEL,

                        "Forced LLM Model":
                            (
                                OLLAMA_FORCED_MODEL
                                or "None"
                            ),

                        "Legacy Default Model":
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

            # Keep raw diagnostics in the console/evidence log, but do not
            # expose low-level connection/timeout stack text to normal users.
            print(f"[REQUEST ERROR] {type(e).__name__}: {e}")
            error_message = _friendly_request_error(e)

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
