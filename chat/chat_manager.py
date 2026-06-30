from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import streamlit as st

# ==========================================================
# CONSTANTS
# ==========================================================

DEFAULT_CHAT_TITLE = "New Chat"
MAX_CHAT_TITLE_LENGTH = 45


class ChatManager:
    """
    ==========================================================
    ChatManager

    Responsible only for conversation management.

    It does NOT render UI.
    It does NOT call the LLM.

    Responsibilities:
        • Create conversations
        • Switch conversations
        • Delete conversations
        • Rename conversations
        • Add messages
        • Return current conversation

    Storage:
        streamlit.session_state
    ==========================================================
    """

    # ======================================================
    # INITIALIZE
    # ======================================================

    @staticmethod
    def initialize():

        if "conversations" not in st.session_state:
            st.session_state.conversations = []

        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = None

        if "current_topic" not in st.session_state:
            st.session_state.current_topic = None

        if len(st.session_state.conversations) == 0:
            ChatManager.create_chat()

    # ======================================================
    # CREATE CHAT
    # ======================================================

    @staticmethod
    def create_chat():

        chat_id = str(uuid4())

        now = datetime.now()

        conversation = {

            "id": chat_id,

            "title": DEFAULT_CHAT_TITLE,

            "created_at": now,

            "updated_at": now,

            "messages": []

        }

        # Newest conversation appears first
        st.session_state.conversations.insert(
            0,
            conversation
        )

        st.session_state.current_chat_id = chat_id

        return conversation

    # ======================================================
    # GET ALL CHATS
    # ======================================================

    @staticmethod
    def get_all_chats():

        return st.session_state.conversations

    # ======================================================
    # GET RECENT CHATS
    # ======================================================

    @staticmethod
    def get_recent_chats():
        """
        Return only conversations that already contain messages.

        Empty placeholder chats are hidden from the Recent Chats list.
        """

        return [

            chat

            for chat in st.session_state.conversations

            if len(chat["messages"]) > 0

        ]

    # ======================================================
    # GET CHAT
    # ======================================================

    @staticmethod
    def get_chat(chat_id):

        for conversation in st.session_state.conversations:

            if conversation["id"] == chat_id:

                return conversation

        return None

    # ======================================================
    # CURRENT CHAT
    # ======================================================

    @staticmethod
    def get_current_chat():

        conversation = ChatManager.get_chat(
            st.session_state.current_chat_id
        )

        if conversation is None:

            conversation = ChatManager.create_chat()

        return conversation

    # ======================================================
    # CURRENT CHAT ID
    # ======================================================

    @staticmethod
    def current_chat_id():

        return st.session_state.current_chat_id

    # ======================================================
    # CURRENT MESSAGES
    # ======================================================

    @staticmethod
    def get_current_messages():

        return ChatManager.get_current_chat()["messages"]

    # ======================================================
    # MESSAGE COUNT
    # ======================================================

    @staticmethod
    def message_count():

        return len(
            ChatManager.get_current_messages()
        )

    # ======================================================
    # HAS MESSAGES
    # ======================================================

    @staticmethod
    def has_messages():

        return ChatManager.message_count() > 0

    # ======================================================
    # IS CURRENT CHAT EMPTY
    # ======================================================

    @staticmethod
    def is_current_chat_empty():

        return ChatManager.message_count() == 0

    # ======================================================
    # ADD MESSAGE
    # ======================================================

    @staticmethod
    def add_message(
        role,
        content,
        sources=None
    ):
        """
        Add a message to the current conversation.

        Parameters
        ----------
        role : str
            "user" or "assistant"

        content : str
            Message content.

        sources : list[str] | None
            Source filenames used to generate the answer.
            Stored with the assistant message so they persist
            after Streamlit reruns.
        """

        conversation = ChatManager.get_current_chat()

        conversation["messages"].append(
            {
                "role": role,
                "content": content,
                "sources": sources or [],
                "timestamp": datetime.now()
            }
        )

        conversation["updated_at"] = datetime.now()

        # Move active conversation to the top
        st.session_state.conversations.remove(
            conversation
        )

        st.session_state.conversations.insert(
            0,
            conversation
        )

        # Auto title from first user message
        if (
            role == "user"
            and conversation["title"] == DEFAULT_CHAT_TITLE
        ):

            conversation["title"] = (
                ChatManager.generate_title(
                    content
                )
            )

    # ======================================================
    # SWITCH CHAT
    # ======================================================

    @staticmethod
    def switch_chat(chat_id):

        conversation = ChatManager.get_chat(
            chat_id
        )

        if conversation is None:

            return False

        st.session_state.current_chat_id = chat_id

        return True

    # ======================================================
    # DELETE CHAT
    # ======================================================

    @staticmethod
    def delete_chat(chat_id):

        st.session_state.conversations = [

            chat

            for chat in st.session_state.conversations

            if chat["id"] != chat_id

        ]

        if len(st.session_state.conversations) == 0:

            ChatManager.create_chat()

            return

        if st.session_state.current_chat_id == chat_id:

            st.session_state.current_chat_id = (
                st.session_state.conversations[0]["id"]
            )

    # ======================================================
    # RENAME CHAT
    # ======================================================

    @staticmethod
    def rename_chat(
        chat_id,
        new_title
    ):

        conversation = ChatManager.get_chat(
            chat_id
        )

        if conversation is None:

            return False

        conversation["title"] = (
            new_title.strip()
            or DEFAULT_CHAT_TITLE
        )

        conversation["updated_at"] = datetime.now()

        return True

    # ======================================================
    # AUTO TITLE
    # ======================================================

    @staticmethod
    def generate_title(question):

        question = question.strip()

        if not question:

            return DEFAULT_CHAT_TITLE

        if len(question) > MAX_CHAT_TITLE_LENGTH:

            return (
                question[:MAX_CHAT_TITLE_LENGTH]
                + "..."
            )

        return question
    
    # ======================================================
    # CURRENT TOPIC
    # ======================================================

    @staticmethod
    def get_current_topic():

        return st.session_state.current_topic

    #@staticmethod
    #def set_current_topic(
    #    topic
    #):

    #    st.session_state.current_topic = topic

    @staticmethod
    def set_current_topic(
        topic
    ):

        st.session_state.current_topic = (
            topic.strip()
            if topic
            else None
        )

    @staticmethod
    def clear_current_topic():

        st.session_state.current_topic = None