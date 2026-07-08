import streamlit as st
import base64
from pathlib import Path
#from urllib.parse import quote
import os

from config.settings import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT
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
    # SESSION INITIALIZATION
    # =====================================================

    @staticmethod
    def initialize_session():

        # Reserved for future UI state
        # Uncomment when implementing features such as:
        #   • Sidebar search
        #   • Theme switch
        #   • Sidebar collapse

        #defaults = {
        #    "sidebar_search": "",
        #    "theme": "dark"
        #}
        #
        #for key, value in defaults.items():
        #    if key not in st.session_state:
        #        st.session_state[key] = value

        pass

    # =====================================================
    # SIDEBAR
    # =====================================================

    @staticmethod
    def render_sidebar():

        with st.sidebar:

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

            # Push footer to bottom
            st.markdown(
                "<div class='sidebar-spacer'></div>",
                unsafe_allow_html=True
            )

#            st.markdown("---")

#            st.html("""
#                <div class="sidebar-footer">
#                    <strong>DocuBot</strong><br>
#                    v1.0
#                </div>
#            """)

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

                    st.markdown(content)

                    if sources:
                        StreamlitUI.render_sources(sources, message_index)

                else:

                    st.markdown(content)

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
    # SOURCES
    # =====================================================

    @staticmethod
    def render_sources(sources, message_index=0):

        if not sources:
            return

        unique_sources = []
        seen = set()

        for source in sources:

            key = source["path"]

            if key not in seen:
                seen.add(key)
                unique_sources.append(source)

        #st.markdown(
        #    '<div class="docubot-sources">',
        #    unsafe_allow_html=True
        #)

        st.markdown("#### 📄 Source")

        for i, source in enumerate(unique_sources):

            if st.button(
                f"📄 {source['name']}",
                key=f"source_{message_index}_{i}"
            ):
                os.startfile(source["path"])

        #st.markdown(
        #    "</div>",
        #    unsafe_allow_html=True
        #)

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