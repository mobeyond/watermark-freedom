import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Import configuration
try:
    import config
except ImportError:
    print("Error: Configuration file (config.py) not found or contains errors.")
    print("Please ensure config.py exists in the telegram_agent directory and defines TELEGRAM_TOKEN.")
    exit()

# Enable logging (optional, but good for development)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    await update.message.reply_text('Hello! I am your new Telegram agent. How can I help you today?')
    logger.info(f"User {update.effective_user.id} started the bot.")

async def send_message_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message to a specified chat ID.
    Usage: /send <chat_id> <message_text>
    """
    try:
        # args will be a list of strings from the command, e.g., ['chat_id', 'message', 'part', '2']
        target_chat_id = context.args[0]
        message_text = " ".join(context.args[1:])

        if not target_chat_id.lstrip('-').isdigit(): # Check if chat_id is numeric (can be negative for groups)
            await update.message.reply_text("Error: Invalid Chat ID. It should be a number.")
            logger.warning(f"Invalid chat_id format from user {update.effective_user.id}: {target_chat_id}")
            return

        if not message_text:
            await update.message.reply_text("Error: Message text cannot be empty. Usage: /send <chat_id> <message>")
            return

        await context.bot.send_message(chat_id=target_chat_id, text=message_text)
        await update.message.reply_text(f"Message sent to {target_chat_id}.")
        logger.info(f"User {update.effective_user.id} sent message to {target_chat_id}: {message_text}")

    except IndexError:
        await update.message.reply_text("Usage: /send <chat_id> <message_text>")
        logger.warning(f"User {update.effective_user.id} used /send command with insufficient arguments.")
    except Exception as e:
        logger.error(f"Error sending message via /send command by user {update.effective_user.id}: {e}")
        await update.message.reply_text(f"An error occurred: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Logs and echoes incoming text messages that are not commands.""" # Docstring updated
    user = update.effective_user
    message_text = update.message.text

    logger.info(f"Received message from {user.id} ({user.username}): \"{message_text}\"")

    # Echo the message back to the user
    await update.message.reply_text(message_text)
    logger.info(f"Echoed message back to {user.id} ({user.username}).") # Optional: log the echo action

def main() -> None:
    """Start the bot."""
    logger.info("Starting bot...")

    if not config.TELEGRAM_TOKEN or config.TELEGRAM_TOKEN == 'YOUR_TELEGRAM_API_TOKEN':
        logger.error("Telegram API token is not configured. "
                     "Please replace 'YOUR_TELEGRAM_API_TOKEN' in config.py with your actual token.")
        print("CRITICAL: Telegram API token not configured in telegram_agent/config.py. Exiting.")
        return

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(config.TELEGRAM_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("send", send_message_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    logger.info("Bot application created. Starting polling...")
    print("Bot is running. Press Ctrl-C to stop.")
    application.run_polling()
    logger.info("Bot stopped.")

if __name__ == '__main__':
    main()
