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

            "messages": [],

            "current_topic": None

        }

        # Newest conversation appears first
        st.session_state.conversations.insert(
            0,
            conversation
        )

        st.session_state.current_chat_id = chat_id
        st.session_state.current_topic = None

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
    # PREPARE REGENERATION
    # ======================================================

    @staticmethod
    def get_regeneration_question(
        message_index
    ):
        """
        Return the user question linked to one assistant response.

        This method does not delete or modify any message.
        """

        messages = ChatManager.get_current_messages()

        if not isinstance(
            message_index,
            int
        ):

            return None

        if (
            message_index < 0
            or message_index >= len(messages)
        ):

            return None

        selected_message = messages[
            message_index
        ]

        if selected_message.get("role") != "assistant":

            return None

        for index in range(
            message_index - 1,
            -1,
            -1
        ):

            previous_message = messages[index]

            if previous_message.get("role") == "user":

                question = (
                    previous_message
                    .get(
                        "content",
                        ""
                    )
                    .strip()
                )

                return question or None

        return None

    # ======================================================
    # BEGIN REGENERATION CONTEXT
    # ======================================================

    @staticmethod
    def begin_regeneration(
        message_index,
        chat_id
    ):
        """
        Temporarily expose only the conversation context that existed
        before the selected assistant response.

        The full conversation is restored after answer generation.
        """

        if chat_id != ChatManager.current_chat_id():

            return None

        conversation = ChatManager.get_current_chat()

        messages = conversation.get(
            "messages",
            []
        )

        question = ChatManager.get_regeneration_question(
            message_index
        )

        if not question:

            return None

        original_messages = messages
        original_topic = conversation.get(
            "current_topic",
            None
        )
        original_session_topic = st.session_state.get(
            "current_topic",
            None
        )

        snapshot = {
            "chat_id": chat_id,
            "message_index": message_index,
            "question": question,
            "messages": original_messages,
            "conversation_topic": original_topic,
            "session_topic": original_session_topic
        }

        # Keep messages only up to the user question that produced
        # the selected assistant response. This prevents later turns
        # from contaminating the regenerated answer.
        conversation["messages"] = (
            original_messages[:message_index]
        )

        conversation["current_topic"] = None
        st.session_state.current_topic = None

        return snapshot

    # ======================================================
    # COMPLETE REGENERATION
    # ======================================================

    @staticmethod
    def complete_regeneration(
        snapshot,
        answer,
        sources=None
    ):
        """
        Replace only the selected assistant response.

        No other message is removed or changed.
        """

        if not snapshot:

            return False

        conversation = ChatManager.get_chat(
            snapshot.get(
                "chat_id"
            )
        )

        if conversation is None:

            return False

        original_messages = snapshot.get(
            "messages",
            []
        )

        message_index = snapshot.get(
            "message_index"
        )

        conversation["messages"] = (
            original_messages
        )

        if (
            not isinstance(message_index, int)
            or message_index < 0
            or message_index >= len(original_messages)
        ):

            ChatManager.restore_regeneration(
                snapshot
            )

            return False

        target_message = original_messages[
            message_index
        ]

        if target_message.get("role") != "assistant":

            ChatManager.restore_regeneration(
                snapshot
            )

            return False

        target_message["content"] = answer
        target_message["sources"] = sources or []
        target_message["timestamp"] = datetime.now()

        conversation["updated_at"] = datetime.now()

        # Preserve the topic state of the complete conversation,
        # because later messages remain in place.
        conversation["current_topic"] = snapshot.get(
            "conversation_topic"
        )

        st.session_state.current_topic = snapshot.get(
            "session_topic"
        )

        return True

    # ======================================================
    # RESTORE REGENERATION
    # ======================================================

    @staticmethod
    def restore_regeneration(
        snapshot
    ):
        """
        Restore the untouched conversation when regeneration fails.
        """

        if not snapshot:

            return False

        conversation = ChatManager.get_chat(
            snapshot.get(
                "chat_id"
            )
        )

        if conversation is None:

            return False

        conversation["messages"] = snapshot.get(
            "messages",
            []
        )

        conversation["current_topic"] = snapshot.get(
            "conversation_topic"
        )

        st.session_state.current_topic = snapshot.get(
            "session_topic"
        )

        return True

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

        st.session_state.current_topic = conversation.get(
            "current_topic",
            None
        )

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

            next_conversation = st.session_state.conversations[0]

            st.session_state.current_chat_id = (
                next_conversation["id"]
            )

            st.session_state.current_topic = (
                next_conversation.get(
                    "current_topic",
                    None
                )
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

        clean_topic = (
            topic.strip()
            if topic
            else None
        )

        st.session_state.current_topic = clean_topic

        conversation = ChatManager.get_current_chat()

        if conversation is not None:

            conversation["current_topic"] = clean_topic

    @staticmethod
    def clear_current_topic():

        st.session_state.current_topic = None

        conversation = ChatManager.get_current_chat()

        if conversation is not None:

            conversation["current_topic"] = None