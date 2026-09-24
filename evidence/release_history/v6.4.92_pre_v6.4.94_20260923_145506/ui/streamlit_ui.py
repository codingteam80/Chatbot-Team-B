import streamlit as st
import streamlit.components.v1 as components
import base64
import html
import mimetypes
import re
from pathlib import Path
#from urllib.parse import quote
import os

from config.settings import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    LAN_SERVER_MODE
)

from chat.chat_manager import ChatManager


class StreamlitUI:

    # =====================================================
    # LOAD HERO LOGO
    # =====================================================

    @staticmethod
    def assoc_logo():

        logo_path = Path("assets_logos/Logo_TSUKIDEN.png")

        if not logo_path.exists():
            return ""

        with open(logo_path, "rb") as image:

            return base64.b64encode(
                image.read()
            ).decode()
        
    # =====================================================
    # PAGE CONFIGURATION
    # =====================================================

    @staticmethod
    def configure():

        st.set_page_config(
            page_title=PAGE_TITLE,
            page_icon=PAGE_ICON,
            layout=LAYOUT,
            initial_sidebar_state="expanded"
        )

        css_file = Path(__file__).parent / "styles.css"

        if css_file.exists():

            with open(
                css_file,
                "r",
                encoding="utf-8"
            ) as css:

                st.markdown(
                    f"<style>{css.read()}</style>",
                    unsafe_allow_html=True
                )

    # =====================================================
    # CHAT INPUT KEYBOARD BEHAVIOR
    # =====================================================

    @staticmethod
    def install_chat_input_keyboard_behavior():
        """Make the multiline chat editor behave like a modern chat input.

        Enter submits the surrounding Streamlit form. Shift+Enter keeps the
        native textarea newline behavior. The handler also grows the editor
        up to a compact maximum height so pasted C/C++ snippets stay readable.
        """

        components.html(
            r"""
            <script>
            (() => {
                let parentWindow;
                let parentDocument;
                try {
                    parentWindow = window.parent;
                    parentDocument = parentWindow.document;
                } catch (error) {
                    return;
                }

                const observerKey = "__docubotChatInputObserver";
                if (parentWindow[observerKey]) {
                    try { parentWindow[observerKey].disconnect(); } catch (error) {}
                }

                const selector = [
                    '.st-key-chat_input_shell_empty [data-testid="stTextArea"] textarea',
                    '.st-key-chat_input_shell_active [data-testid="stTextArea"] textarea'
                ].join(',');

                function bindEditor() {
                    const textarea = parentDocument.querySelector(selector);
                    if (!textarea) return;

                    const shell = textarea.closest(
                        '.st-key-chat_input_shell_empty, .st-key-chat_input_shell_active'
                    );

                    const applyComposerHeight = (editorHeight) => {
                        textarea.style.setProperty(
                            'height',
                            `${editorHeight}px`,
                            'important'
                        );

                        if (shell) {
                            const toolbarHeight = 46;
                            shell.style.setProperty(
                                '--docubot-editor-height',
                                `${editorHeight}px`
                            );
                            const composerHeight = editorHeight + toolbarHeight;
                            shell.style.setProperty(
                                '--docubot-composer-height',
                                `${composerHeight}px`
                            );
                            parentDocument.documentElement.style.setProperty(
                                '--docubot-active-composer-height',
                                `${composerHeight}px`
                            );
                        }
                    };

                    const resetCompact = () => {
                        // ChatGPT-like contract: once a message is submitted
                        // (or Streamlit clears the draft on rerun), the composer
                        // immediately returns to its compact one-line height.
                        applyComposerHeight(48);
                    };

                    const resize = () => {
                        // An empty draft must never inherit the height of the
                        // previously submitted multiline message.  React can
                        // update textarea.value without firing a DOM input
                        // event, so check the current value every time bindEditor
                        // runs as well as on normal input events.
                        if (!String(textarea.value || '').length) {
                            resetCompact();
                            return;
                        }

                        // CSS uses the same variables.  Set the inline height
                        // with !important while measuring so an older cached
                        // stylesheet cannot pin the editor to one line.
                        textarea.style.setProperty('height', 'auto', 'important');
                        const editorHeight = Math.max(
                            48,
                            Math.min(180, textarea.scrollHeight)
                        );
                        applyComposerHeight(editorHeight);
                    };

                    if (textarea.dataset.docubotKeyboardBound !== '1') {
                        textarea.dataset.docubotKeyboardBound = '1';

                        textarea.addEventListener('keydown', (event) => {
                            if (event.key !== 'Enter' || event.isComposing) return;

                            // Shift+Enter intentionally uses the textarea's
                            // native newline behavior.
                            if (event.shiftKey) return;

                            event.preventDefault();
                            event.stopPropagation();

                            const currentShell = textarea.closest(
                                '.st-key-chat_input_shell_empty, .st-key-chat_input_shell_active'
                            );
                            const submit = currentShell
                                ? currentShell.querySelector(
                                    '.st-key-chat_send_button [data-testid="stFormSubmitButton"] button'
                                  )
                                : null;

                            if (submit && !submit.disabled) {
                                // Reset before React/Streamlit reruns so the
                                // large multiline box does not remain visible
                                // during "Searching knowledge base...".
                                resetCompact();
                                submit.click();
                            }
                        }, true);

                        textarea.addEventListener('input', resize);
                    }

                    const submitButton = shell
                        ? shell.querySelector(
                            '.st-key-chat_send_button [data-testid="stFormSubmitButton"] button'
                          )
                        : null;
                    if (submitButton && submitButton.dataset.docubotResetBound !== '1') {
                        submitButton.dataset.docubotResetBound = '1';
                        submitButton.addEventListener('click', resetCompact, true);
                    }

                    resize();
                }

                const observer = new MutationObserver(bindEditor);
                observer.observe(parentDocument.body, {
                    childList: true,
                    subtree: true
                });
                parentWindow[observerKey] = observer;

                bindEditor();
                window.setTimeout(bindEditor, 80);
                window.setTimeout(bindEditor, 300);
            })();
            </script>
            """,
            height=0,
            scrolling=False,
        )

    # =====================================================
    # SESSION INITIALIZATION
    # =====================================================

    @staticmethod
    def initialize_session():

        # UI state for controlled chat submission.
        #
        # The textbox remains editable while DocuBot
        # is generating an answer, but the send action
        # is blocked until processing is finished.
        defaults = {
            "is_processing": False,
            "pending_question": None,
            "chat_draft": "",
            "clear_chat_draft": False,
            "regenerate_message_index": None,
            "regenerate_chat_id": None
        }

        for key, value in defaults.items():

            if key not in st.session_state:

                st.session_state[key] = value

    # =====================================================
    # SIDEBAR
    # =====================================================

    @staticmethod
    def render_sidebar():

        with st.sidebar:

            # The sidebar is intentionally split into two independent
            # regions.  The upper controls stay fixed while only the
            # recent-chat list is allowed to scroll.
            with st.container(key="sidebar_fixed_top"):

                # ---------------------------------------------
                # COMPANY LOGO
                # ---------------------------------------------

                logo = StreamlitUI.assoc_logo()

                if logo:

                    st.markdown(
                        f"""
                        <div class="company-logo-wrapper">
                            <img
                                src="data:image/png;base64,{logo}"
                                class="company-logo"
                            >
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ---------------------------------------------
                # DOCUBOT TITLE
                # ---------------------------------------------

                st.markdown(
                    """
                    <div class="sidebar-logo">
                        🤖 DocuBot
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <div class="sidebar-title">
                        Your company's knowledge assistant
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <div class="sidebar-version">
                        v1.0
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                # ---------------------------------------------
                # NEW CHAT
                # ---------------------------------------------

                if st.button(
                    "✚ New Chat",
                    use_container_width=True
                ):

                    ChatManager.create_chat()

                    StreamlitUI.rerun()

                st.write("")

            # Keep the divider and section label at the original v6.4.9
            # inset while only the cards below them scroll.
            with st.container(key="recent_chats_header"):
                st.markdown("---")

                st.markdown(
                    """
                    <div class="sidebar-section">
                        Recent Chats
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            conversations = ChatManager.get_recent_chats()

            with st.container(key="recent_chats_scroll"):

                # Keep the conversation cards at the exact v6.4.9 inset/width
                # while the parent scroll track is allowed to reach the
                # sidebar's outer-right edge.
                with st.container(key="recent_chats_cards"):

                    if not conversations:
                        st.button(
                            "No recent chats",
                            disabled=True,
                            use_container_width=True
                        )

                    else:

                        for conversation in conversations:

                            is_current = (

                                conversation["id"]

                                ==

                                ChatManager.current_chat_id()

                            )

                            button_type = (

                                "primary"

                                if is_current

                                else

                                "secondary"

                            )

                            if st.button(

                                f"💬 {conversation['title']}",

                                key=conversation["id"],

                                use_container_width=True,

                                type=button_type

                            ):

                                ChatManager.switch_chat(
                                    conversation["id"]
                                )

                                StreamlitUI.rerun()

    # =====================================================
    # WELCOME SCREEN
    # =====================================================

    @staticmethod
    def show_welcome():

        if not ChatManager.is_current_chat_empty():
            return

        st.html("""
        <div class="hero-container">

            <div class="hero-title">
                Hi! Is there anything I can help you with?
            </div>

            <div class="hero-description">
                Search company policies, manuals,
                standards, procedures and internal
                company knowledge.
            </div>

        </div>
        """)

    # =====================================================
    # CHAT HISTORY
    # =====================================================

    @staticmethod
    def render_chat_history():

        messages = ChatManager.get_current_messages()

        if not messages:
            return

        active_regeneration_index = None

        if (
            st.session_state.is_processing
            and st.session_state.regenerate_chat_id
            == ChatManager.current_chat_id()
        ):

            active_regeneration_index = (
                st.session_state.regenerate_message_index
            )

        for message_index, message in enumerate(messages):

            role = message.get(
                "role",
                "assistant"
            )

            content = message.get(
                "content",
                ""
            )

            sources = message.get(
                "sources",
                []
            )

            avatar = (
                "☺"
                if role == "user"
                else "🤖"
            )

            with st.chat_message(
                role,
                avatar=avatar
            ):

                if role == "assistant":

                    is_regenerating_this_response = (
                        message_index
                        == active_regeneration_index
                    )

                    if is_regenerating_this_response:

                        st.markdown(
                            """
                            <div class="loading-bubble">
                                <div class="loading-spinner"></div>
                                <div>Searching knowledge base...</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(content)

                        if sources:

                            StreamlitUI.render_sources(
                                sources,
                                message_index
                            )

                        # Keep response actions on one horizontal row.
                        StreamlitUI.render_response_actions(
                            content=content,
                            message_index=message_index,
                            allow_regenerate=True
                        )

                else:

                    StreamlitUI.render_user_message(content)

    # =====================================================
    # USER MESSAGE PRESENTATION
    # =====================================================

    @staticmethod
    def render_user_message(content: str):
        """Render the submitted text exactly as typed, including newlines.

        User content is HTML-escaped first, then displayed with a pre-wrap
        container.  This prevents Markdown from collapsing pasted C/C++ code
        or treating source characters as formatting syntax.
        """
        safe_content = html.escape(str(content or ""))
        st.markdown(
            f'<div class="docubot-user-message-text">{safe_content}</div>',
            unsafe_allow_html=True,
        )

    # =====================================================
    # BUILD ASSISTANT MESSAGE
    # =====================================================

#    @staticmethod
#    def build_assistant_message(answer, sources):

#        html = f"""
#        <div class="assistant-answer">

#        <div class="assistant-text">
#        {answer}
#        </div>
#        """

#        if sources:

#            unique_sources = list(dict.fromkeys(sources))

#            title = (
#                "📄 Source"
#                if len(unique_sources) == 1
#                else f"📄 Sources ({len(unique_sources)})"
#            )

#            html += """
#                <hr class="sources-divider">
#            """

#            html += f"""
#                <div class="sources-title">
#                    {title}
#                </div>
#            """

#            for source in unique_sources:

#                html += f"""
#                    <div class="source-item">
#                        {source}
#                    </div>
#                """

#        html += """
#        </div>
#        """

#        return html

    # =====================================================
    # RESPONSE ACTIONS
    # =====================================================

    @staticmethod
    def render_response_actions(
        content,
        message_index,
        allow_regenerate=False
    ):
        """
        Render Copy and Regenerate on one horizontal row.

        Real Streamlit columns are used so the actions cannot
        fall into a vertical stack.
        """

        with st.container(
            key=f"docubot_actions_{message_index}"
        ):

            if allow_regenerate:

                copy_column, regenerate_column = st.columns(
                    [74, 108],
                    gap="small"
                )

                with copy_column:

                    StreamlitUI.render_copy_action(
                        content,
                        message_index
                    )

                with regenerate_column:

                    StreamlitUI.render_regenerate_action(
                        message_index
                    )

            else:

                StreamlitUI.render_copy_action(
                    content,
                    message_index
                )

    # =====================================================
    # COPY RESPONSE
    # =====================================================

    @staticmethod
    def render_copy_action(
        content,
        message_index=0
    ):
        """
        Render a client-side Copy button for an assistant answer.

        Only the answer text is copied. Sources and file paths
        are not included. Copying does not rerun the Streamlit app.
        """

        if not content:
            return

        message_key = (
            "live"
            if message_index < 0
            else str(message_index)
        )

        encoded_content = base64.b64encode(
            content.encode("utf-8")
        ).decode("ascii")

        component_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                html,
                body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    overflow: hidden;
                    font-family:
                        Inter,
                        system-ui,
                        -apple-system,
                        BlinkMacSystemFont,
                        "Segoe UI",
                        sans-serif;
                }}

                .copy-button {{
                    height: 26px;
                    padding: 3px 8px;

                    display: inline-flex;
                    align-items: center;
                    gap: 5px;

                    background: transparent;
                    color: #8d98aa;

                    border: 1px solid transparent;
                    border-radius: 6px;

                    font-size: 11px;
                    font-weight: 500;
                    line-height: 1;

                    cursor: pointer;

                    transition:
                        color .16s ease,
                        background .16s ease,
                        border-color .16s ease;
                }}

                .copy-button:hover {{
                    background: rgba(78, 161, 255, .10);
                    color: #9cc9ff;
                    border-color: rgba(78, 161, 255, .20);
                }}

                .copy-button:focus,
                .copy-button:active {{
                    outline: none;
                    box-shadow: none;
                }}

                .copy-button.copied {{
                    color: #9cc9ff;
                }}
            </style>
        </head>

        <body>
            <button
                id="copy-button"
                class="copy-button"
                type="button"
                aria-label="Copy response"
                title="Copy response"
            >
                <span id="copy-icon">⧉</span>
                <span id="copy-label">Copy</span>
            </button>

            <script>
                const encodedContent = "{encoded_content}";

                const decodedBytes = Uint8Array.from(
                    atob(encodedContent),
                    character => character.charCodeAt(0)
                );

                const answerText = new TextDecoder(
                    "utf-8"
                ).decode(decodedBytes);

                const copyButton = document.getElementById(
                    "copy-button"
                );

                const copyIcon = document.getElementById(
                    "copy-icon"
                );

                const copyLabel = document.getElementById(
                    "copy-label"
                );

                async function copyWithFallback(text) {{
                    try {{
                        await navigator.clipboard.writeText(text);
                        return true;
                    }}
                    catch (clipboardError) {{
                        const textArea = document.createElement(
                            "textarea"
                        );

                        textArea.value = text;
                        textArea.setAttribute(
                            "readonly",
                            ""
                        );

                        textArea.style.position = "fixed";
                        textArea.style.opacity = "0";
                        textArea.style.pointerEvents = "none";

                        document.body.appendChild(textArea);

                        textArea.focus();
                        textArea.select();

                        const copied = document.execCommand(
                            "copy"
                        );

                        document.body.removeChild(textArea);

                        return copied;
                    }}
                }}

                copyButton.addEventListener(
                    "click",
                    async () => {{
                        const copied = await copyWithFallback(
                            answerText
                        );

                        if (!copied) {{
                            copyLabel.textContent = "Copy failed";
                            return;
                        }}

                        copyButton.classList.add("copied");
                        copyIcon.textContent = "✓";
                        copyLabel.textContent = "Copied";

                        window.setTimeout(
                            () => {{
                                copyButton.classList.remove(
                                    "copied"
                                );

                                copyIcon.textContent = "⧉";
                                copyLabel.textContent = "Copy";
                            }},
                            1400
                        );
                    }}
                );
            </script>
        </body>
        </html>
        """

        with st.container(
            key=f"docubot_copy_{message_key}"
        ):

            components.html(
                component_html,
                height=28,
                scrolling=False
            )

    # =====================================================
    # REGENERATE RESPONSE
    # =====================================================

    @staticmethod
    def render_regenerate_action(
        message_index
    ):
        """
        Render a Regenerate button for the latest assistant reply.

        Only the selected assistant reply is regenerated in place.
        Other earlier and later conversation messages remain unchanged.
        """

        with st.container(
            key=f"docubot_regenerate_{message_index}"
        ):

            regenerate_clicked = st.button(
                "↻ Regenerate",
                key=f"regenerate_button_{message_index}",
                disabled=st.session_state.is_processing
            )

        if not regenerate_clicked:
            return

        question = ChatManager.get_regeneration_question(
            message_index
        )

        if not question:
            return

        st.session_state.is_processing = True
        st.session_state.pending_question = question
        st.session_state.regenerate_message_index = (
            message_index
        )
        st.session_state.regenerate_chat_id = (
            ChatManager.current_chat_id()
        )
        st.session_state.clear_chat_draft = False

        StreamlitUI.rerun()

    # =====================================================
    # SOURCES
    # =====================================================

    @staticmethod
    def _compact_reference_label(references):
        refs = []
        for reference in references or []:
            value = str(reference or "").strip()
            if value and value not in refs:
                refs.append(value)

        if not refs:
            return ""
        if len(refs) == 1:
            return refs[0]

        parsed = []
        for value in refs:
            match = re.fullmatch(
                r"(?i)(Rule|Dir(?:ective)?)\s+(\d+)\.(\d+)",
                value,
            )
            if not match:
                parsed = []
                break
            kind = "Directive" if match.group(1).casefold().startswith("dir") else "Rule"
            parsed.append((kind, int(match.group(2)), int(match.group(3))))

        if parsed:
            kinds = {item[0] for item in parsed}
            majors = {item[1] for item in parsed}
            minors = sorted({item[2] for item in parsed})
            if len(kinds) == 1 and len(majors) == 1 and minors:
                consecutive = minors == list(range(minors[0], minors[-1] + 1))
                kind = next(iter(kinds))
                major = next(iter(majors))
                if consecutive:
                    plural = "Directives" if kind == "Directive" else "Rules"
                    return f"{plural} {major}.{minors[0]}–{major}.{minors[-1]}"

        if len(refs) <= 3:
            return ", ".join(refs)
        return f"{len(refs)} references"

    @staticmethod
    def _aggregate_sources(sources):
        """Merge repeated chunks from one file into one compact source chip."""

        grouped = {}
        order = []
        for source in sources or []:
            source_path = str(source.get("path") or "").strip()
            if not source_path:
                continue

            if source_path not in grouped:
                grouped[source_path] = {
                    "name": source.get("name") or Path(source_path).name or "Source file",
                    "path": source_path,
                    "page_starts": [],
                    "page_ends": [],
                    "references": [],
                }
                order.append(source_path)

            item = grouped[source_path]
            for key, target in (("page_start", "page_starts"), ("page_end", "page_ends")):
                value = source.get(key)
                try:
                    numeric = int(value) if value is not None and str(value).strip() else None
                except (TypeError, ValueError):
                    numeric = None
                if numeric is not None:
                    item[target].append(numeric)

            reference = str(source.get("reference") or "").strip()
            if reference and reference not in item["references"]:
                item["references"].append(reference)

        output = []
        for source_path in order:
            item = grouped[source_path]
            starts = item.pop("page_starts")
            ends = item.pop("page_ends")
            references = item.pop("references")
            pages = starts + ends
            item["page_start"] = min(pages) if pages else None
            item["page_end"] = max(pages) if pages else None
            item["reference"] = StreamlitUI._compact_reference_label(references)
            output.append(item)

        return output

    @staticmethod
    def render_sources(sources, message_index=0):

        if not sources:
            return

        unique_sources = StreamlitUI._aggregate_sources(sources)

        if not unique_sources:
            return

        source_count = len(unique_sources)

        source_label = (
            "Source"
            if source_count == 1
            else f"Sources · {source_count}"
        )

        message_key = (
            "live"
            if message_index < 0
            else str(message_index)
        )

        sources_container = st.container(
            key=f"docubot_sources_{message_key}"
        )

        sources_container.markdown(
                f"""
                <div class="docubot-sources-heading">
                    {source_label}
                </div>
                """,
                unsafe_allow_html=True
            )

        for source_index, source in enumerate(
                unique_sources
        ):

            source_chip = sources_container.container(
                    key=(
                        "docubot_source_chip_"
                        f"{message_key}_{source_index}"
                    )
                )

            details = []
            page_start = source.get("page_start")
            page_end = source.get("page_end")
            if page_start:
                if page_end and str(page_end) != str(page_start):
                    details.append(f"Pages {page_start}–{page_end}")
                else:
                    details.append(f"Page {page_start}")
            if source.get("reference"):
                details.append(str(source["reference"]))
            source_label_text = source["name"]
            if details:
                source_label_text += " · " + " · ".join(details)

            source_path = Path(source["path"])

            if LAN_SERVER_MODE:
                # A browser on another LAN PC cannot open a filesystem path
                # that exists only on the DocuBot server. Serve the cited file
                # through the existing Streamlit connection instead. This
                # keeps company documents on the central server and avoids
                # exposing a writable/shared Chroma or data directory.
                try:
                    source_bytes = source_path.read_bytes()
                    mime_type = (
                        mimetypes.guess_type(source_path.name)[0]
                        or "application/octet-stream"
                    )
                    source_chip.download_button(
                        f"📄 {source_label_text}",
                        data=source_bytes,
                        file_name=source_path.name,
                        mime=mime_type,
                        key=(
                            "source_download_"
                            f"{message_key}_{source_index}"
                        ),
                    )
                except (OSError, PermissionError):
                    source_chip.button(
                        f"📄 {source_label_text} (unavailable)",
                        key=(
                            "source_unavailable_"
                            f"{message_key}_{source_index}"
                        ),
                        disabled=True,
                    )
            else:
                source_clicked = source_chip.button(
                    f"📄 {source_label_text}",
                    key=(
                        "source_button_"
                        f"{message_key}_{source_index}"
                    )
                )

                if source_clicked:
                    os.startfile(source["path"])

    # =======================
    # ==============================
    # FOOTER
    # =====================================================

    @staticmethod
    def render_footer_note():

        st.markdown(
            """
            <div class="chat-footer">
                DocuBot can make mistakes. Check important information.
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # UI RERUN
    # =====================================================

    @staticmethod
    def rerun():
        """
        Centralized UI refresh.

        Using a helper instead of calling st.rerun()
        directly makes future maintenance easier.

        Example:
            StreamlitUI.rerun()
        """

        st.rerun()