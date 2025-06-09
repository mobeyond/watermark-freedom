import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

# Adjust the import path based on your project structure
# This assumes agent.py is in the parent directory of tests/
import sys
import os
# Add the parent directory (telegram_agent) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import start, handle_message, send_message_command
# Make sure agent.py can be imported. You might need to adjust sys.path or structure.
# For simplicity, we assume TELEGRAM_TOKEN will be patched or not directly used in these unit tests.

from telegram import Update, User, Message, Chat
from telegram.ext import ContextTypes

# A helper function to run async test methods
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

class TestAgentHandlers(unittest.TestCase):

    @async_test
    async def test_start_command(self):
        """Test the /start command handler."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345
        update.message = MagicMock(spec=Message)
        update.message.reply_text = AsyncMock() # Use AsyncMock for async methods

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        await start(update, context)

        update.message.reply_text.assert_called_once_with(
            'Hello! I am your new Telegram agent. How can I help you today?'
        )
        # Check if logger was called (optional, requires logger patching)
        # agent.logger.info.assert_any_call(f"User {update.effective_user.id} started the bot.")


    @async_test
    async def test_handle_message_echo(self):
        """Test the message handler for echoing messages."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345
        update.effective_user.username = "testuser"

        update.message = MagicMock(spec=Message)
        test_message_text = "Hello, bot!"
        update.message.text = test_message_text
        update.message.reply_text = AsyncMock()

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        # Patch logger within agent module if you want to assert log calls
        with patch('agent.logger') as mock_logger:
            await handle_message(update, context)

        update.message.reply_text.assert_called_once_with(test_message_text)
        mock_logger.info.assert_any_call(
            f"Received message from {update.effective_user.id} ({update.effective_user.username}): \"{test_message_text}\""
        )
        mock_logger.info.assert_any_call(
            f"Echoed message back to {update.effective_user.id} ({update.effective_user.username})."
        )

    @async_test
    async def test_send_message_command_success(self):
        """Test the /send command handler for successful sending."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 123
        update.message = MagicMock(spec=Message)
        update.message.reply_text = AsyncMock()

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.args = ["987654321", "Test", "message", "to", "send"]
        context.bot = MagicMock()
        context.bot.send_message = AsyncMock() # Mock the bot's send_message method

        # Patch logger within agent module
        with patch('agent.logger') as mock_logger:
            await send_message_command(update, context)

        expected_message = "Test message to send"
        context.bot.send_message.assert_called_once_with(chat_id="987654321", text=expected_message)
        update.message.reply_text.assert_called_once_with("Message sent to 987654321.")
        mock_logger.info.assert_any_call(
            f"User {update.effective_user.id} sent message to 987654321: {expected_message}"
        )

    @async_test
    async def test_send_message_command_no_args(self):
        """Test /send command with no arguments."""
        update = MagicMock(spec=Update)
        update.message = MagicMock(spec=Message)
        update.message.reply_text = AsyncMock()
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.args = [] # No arguments

        with patch('agent.logger') as mock_logger:
            await send_message_command(update, context)

        update.message.reply_text.assert_called_once_with("Usage: /send <chat_id> <message_text>")
        mock_logger.warning.assert_any_call(
             f"User {update.effective_user.id} used /send command with insufficient arguments."
        )

    @async_test
    async def test_send_message_command_invalid_chat_id(self):
        """Test /send command with an invalid chat ID."""
        update = MagicMock(spec=Update)
        update.message = MagicMock(spec=Message)
        update.message.reply_text = AsyncMock()
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.args = ["invalid_chat_id", "some", "message"]

        with patch('agent.logger') as mock_logger:
            await send_message_command(update, context)

        update.message.reply_text.assert_called_once_with("Error: Invalid Chat ID. It should be a number.")
        mock_logger.warning.assert_any_call(
            f"Invalid chat_id format from user {update.effective_user.id}: invalid_chat_id"
        )

if __name__ == '__main__':
    unittest.main()
