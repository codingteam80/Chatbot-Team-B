import json

import streamlit as st
from contextlib import nullcontext

from ui.streamlit_ui import StreamlitUI
from services.answer_service import AnswerService
from chat.chat_manager import ChatManager
from scripts.smart_build import get_update_plan
from scripts.kb_update_runner import launch_kb_update_subprocess
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


def _kb_plan_signature(plan):
    """Stable identity for one source-change set shown in the confirmation UI."""

    changes = (plan or {}).get("changes") or {}
    payload = {
        "mode": str((plan or {}).get("mode") or ""),
        "reason": str((plan or {}).get("reason") or ""),
        "added": sorted(str(x).casefold() for x in (changes.get("added") or [])),
        "updated": sorted(str(x).casefold() for x in (changes.get("updated") or [])),
        "deleted": sorted(str(x).casefold() for x in (changes.get("deleted") or [])),
        "unchanged": sorted(str(x).casefold() for x in (changes.get("unchanged") or [])),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _kb_update_success_message(update_result):
    """Return a compact structured success marker; technical detail stays in logs."""

    plan = (update_result or {}).get("plan") or {}
    changes = plan.get("changes") or {}
    added = len(changes.get("added") or [])
    updated = len(changes.get("updated") or [])
    deleted = len(changes.get("deleted") or [])
    return f"KB_UPDATE_SUCCESS::{added}::{updated}::{deleted}"


def _render_kb_update_flash(flash_message):
    """Render a reliable fixed notification card without affecting page layout."""

    if flash_message.startswith("KB_UPDATE_SUCCESS::"):
        try:
            _, added, updated, deleted = flash_message.split("::", 3)
            added = int(added)
            updated = int(updated)
            deleted = int(deleted)
        except Exception:
            # Fail safely to a concise generic success notification. Technical
            # update evidence remains in logs/result JSON regardless.
            added = updated = deleted = 0

        st.markdown(
            f"""
            <div class="docubot-kb-update-notification" role="status" aria-live="polite">
                <div class="docubot-kb-update-notification__icon" aria-hidden="true">✓</div>
                <div class="docubot-kb-update-notification__body">
                    <div class="docubot-kb-update-notification__title">
                        Knowledge Base is successfully updated.
                    </div>
                    <div class="docubot-kb-update-notification__counts">
                        <span>Added: <strong>{added}</strong></span>
                        <span>Updated: <strong>{updated}</strong></span>
                        <span>Deleted: <strong>{deleted}</strong></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.toast(flash_message, icon="⚠️")


# ======================================
# INITIALIZE
# ======================================

StreamlitUI.configure()

StreamlitUI.initialize_session()

# ======================================
# KNOWLEDGE BASE STATUS / MAINTENANCE UI
# ======================================

# UI contract: maintenance controls (#2) are rendered first, then the complete
# chat/welcome composer (#1). The potentially long worker is intentionally
# deferred until AFTER both regions are rendered. This prevents Streamlit's
# top-to-bottom execution from temporarily removing/moving the chat composer
# while the KB subprocess is running.

for key, default in (
    ("kb_update_in_progress", False),
    ("kb_update_requested", False),
    ("kb_update_confirm_plan", ""),
    ("kb_smart_update_confirm", False),
    ("kb_update_last_result", None),
    ("kb_update_flash", ""),
    ("kb_update_reset_confirmation", False),
):
    if key not in st.session_state:
        st.session_state[key] = default

# Widget keys may only be reset before their widget is instantiated on a run.
# The worker therefore sets this flag at the bottom, and the next rerun applies
# the checkbox reset here before maintenance region #2 is created.
if st.session_state.get("kb_update_reset_confirmation", False):
    st.session_state.kb_smart_update_confirm = False
    st.session_state.kb_update_reset_confirmation = False

try:
    # Re-evaluate the lightweight plan on each Streamlit rerun so source files
    # copied in on another PC/session are detected without restarting DocuBot.
    kb_plan = get_update_plan()
    st.session_state.kb_outdated = kb_plan["mode"] != "noop"
    st.session_state.kb_update_mode = kb_plan["mode"]
    st.session_state.kb_update_reason = kb_plan.get("reason")
    st.session_state.kb_status_error = ""
except Exception as kb_status_error:
    kb_plan = {
        "mode": "status_error",
        "reason": (
            f"Knowledge-base status check failed: "
            f"{type(kb_status_error).__name__}: {kb_status_error}"
        ),
        "changes": {},
    }
    st.session_state.kb_outdated = True
    st.session_state.kb_update_mode = "status_error"
    st.session_state.kb_update_reason = kb_plan["reason"]
    st.session_state.kb_status_error = kb_plan["reason"]

flash_message = str(st.session_state.get("kb_update_flash") or "").strip()
if flash_message:
    # Success uses a DocuBot-owned fixed notification card instead of the
    # native Streamlit toast. This avoids browser/Streamlit clipping while
    # remaining overlay-only and preserving the #1/#2 page layout.
    _render_kb_update_flash(flash_message)
    st.session_state.kb_update_flash = ""

if st.session_state.kb_outdated:
    full_rebuild_required = st.session_state.get("kb_update_mode") == "full_rebuild"

    # In centralized LAN mode, employee browser sessions must never race a
    # knowledge-base write. Server maintenance remains an admin-only BAT action.
    web_kb_update_allowed = (not LAN_SERVER_MODE or ALLOW_WEB_KB_UPDATE)

    with st.container(key="kb_maintenance_panel"):
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
            update_in_progress = bool(st.session_state.get("kb_update_in_progress", False))
            status_error = st.session_state.get("kb_update_mode") == "status_error"

            if status_error:
                st.error(st.session_state.get("kb_update_reason") or "Knowledge-base status check failed.")
                update_confirmed = False
            else:
                changes = kb_plan.get("changes") or {}
                added_count = len(changes.get("added") or [])
                updated_count = len(changes.get("updated") or [])
                deleted_count = len(changes.get("deleted") or [])
                unchanged_count = len(changes.get("unchanged") or [])

                if full_rebuild_required:
                    st.warning(
                        "Knowledge base maintenance is required before new source changes can be used safely. "
                        + (st.session_state.get("kb_update_reason") or "A full rebuild is required.")
                    )
                    confirmation_text = (
                        "I understand that Update Knowledge Base will safely rebuild all source documents this time."
                    )
                else:
                    st.warning(
                        "Knowledge base changes detected. Update Knowledge Base will apply a "
                        "safe transactional incremental update: "
                        f"new={added_count}, modified={updated_count}, deleted={deleted_count}, "
                        f"unchanged/skipped={unchanged_count}. Unchanged files will not be re-embedded."
                    )
                    confirmation_text = (
                        "I understand that Update Knowledge Base will process only new, modified, "
                        "or deleted files and will skip unchanged files."
                    )

                # Require an explicit confirmation for BOTH incremental and full
                # rebuild modes. Reset it automatically when the detected change
                # set changes, so a previous confirmation cannot authorize a new
                # set of source mutations.
                plan_signature = _kb_plan_signature(kb_plan)
                if st.session_state.get("kb_update_confirm_plan") != plan_signature:
                    st.session_state.kb_update_confirm_plan = plan_signature
                    st.session_state.kb_smart_update_confirm = False

                update_confirmed = st.checkbox(
                    confirmation_text,
                    key="kb_smart_update_confirm",
                    disabled=update_in_progress,
                )

            update_clicked = st.button(
                "Update Knowledge Base",
                key="kb_smart_update",
                disabled=(not update_confirmed or update_in_progress),
            )

            last_result = st.session_state.get("kb_update_last_result")
            if (
                not update_in_progress
                and isinstance(last_result, dict)
                and last_result.get("overall") not in {None, "PASS"}
            ):
                stage = str(last_result.get("stage") or "unknown")
                message = str(last_result.get("message") or "Unknown update failure.")
                st.error(f"Knowledge base update stopped at stage '{stage}'. {message}")

            if update_clicked:
                # First rerun paints the disabled maintenance controls and the
                # normal chat/welcome region. The long worker is started only at
                # the bottom of the script on that rerun.
                st.session_state.kb_update_in_progress = True
                st.session_state.kb_update_requested = True
                st.session_state.kb_update_last_result = None
                st.rerun()

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

        kb_chat_disabled = bool(
            st.session_state.get("kb_update_in_progress", False)
        )

        question = st.text_area(
            "Ask company knowledge...",
            key="chat_draft",
            label_visibility="collapsed",
            placeholder=(
                "Knowledge Base update in progress..."
                if kb_chat_disabled
                else "Ask company knowledge..."
            ),
            # Disable the editor itself, not only the send button, while the
            # KB transaction is running. The composer remains in exactly the
            # same position and is re-enabled automatically after the worker.
            disabled=kb_chat_disabled,
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
                disabled=(
                    st.session_state.is_processing
                    or st.session_state.get("kb_update_in_progress", False)
                )
            )

# Browser keyboard contract: Enter sends; Shift+Enter inserts a newline.
# The textarea remains editable while processing, while plain Enter cannot
# submit because the send button is disabled during that state.
StreamlitUI.install_chat_input_keyboard_behavior()

StreamlitUI.render_footer_note()

# ======================================
# DEFERRED KNOWLEDGE-BASE UPDATE WORKER
# ======================================
# Run only after maintenance region #2 AND chat/welcome region #1 have been
# emitted. The browser therefore keeps both regions at their normal positions
# for the entire blocking subprocess call.
if (
    st.session_state.get("kb_update_requested", False)
    and st.session_state.get("kb_update_in_progress", False)
):
    try:
        # No spinner/status row is inserted into normal document flow here.
        # The already-rendered warning + checkbox + disabled Update button keep
        # maintenance region #2 at the same height while the worker runs, so
        # welcome/composer region #1 does not move.
        update_result = launch_kb_update_subprocess(source="streamlit_button")
    except Exception as update_error:
        update_result = {
            "overall": "FAIL",
            "stage": "launcher",
            "message": f"{type(update_error).__name__}: {update_error}",
        }
    finally:
        st.session_state.kb_update_in_progress = False
        st.session_state.kb_update_requested = False

    st.session_state.kb_update_last_result = update_result
    st.session_state.kb_update_reset_confirmation = True

    if update_result.get("overall") == "PASS":
        st.session_state.kb_update_flash = _kb_update_success_message(update_result)
    else:
        stage = str(update_result.get("stage") or "unknown")
        message = str(update_result.get("message") or "Unknown update failure.")
        st.session_state.kb_update_flash = (
            f"Knowledge base update did not complete (stage: {stage}). {message}"
        )

    st.rerun()

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
        and not st.session_state.get("kb_update_in_progress", False)
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
