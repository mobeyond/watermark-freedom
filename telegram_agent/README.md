# Telegram Agent

This is a simple Telegram agent built with Python using the `python-telegram-bot` library.
It can connect to Telegram, receive messages, send messages, and perform basic echo functionality.

## Prerequisites

*   Python 3.7+
*   pip (Python package installer)

## Setup Instructions

1.  **Get a Telegram API Token:**
    *   Open Telegram and search for "BotFather".
    *   Start a chat with BotFather and send the `/newbot` command.
    *   Follow the instructions to choose a name and username for your bot.
    *   BotFather will provide you with an API token. Keep this token secure.

2.  **Configure the Agent:**
    *   Open the `telegram_agent/config.py` file.
    *   Replace `'YOUR_TELEGRAM_API_TOKEN'` with the actual API token you received from BotFather:
        ```python
        TELEGRAM_TOKEN = 'YOUR_ACTUAL_API_TOKEN_HERE'
        ```
    *   Save the `config.py` file.

3.  **Install Dependencies:**
    *   Navigate to the `telegram_agent` directory in your terminal.
    *   Install the required Python libraries by running:
        ```bash
        pip install -r requirements.txt
        ```

## Running the Agent

1.  Ensure you have completed the setup instructions (configured the token and installed dependencies).
2.  Navigate to the `telegram_agent` directory in your terminal.
3.  Run the agent using the following command:
    ```bash
    python agent.py
    ```
4.  You should see log messages indicating the bot is running (e.g., "Bot is running. Press Ctrl-C to stop.").
5.  Open Telegram and interact with your bot.

## Available Commands

*   `/start`: Sends a welcome message.
*   `/send <chat_id> <message_text>`: Instructs the bot to send `<message_text>` to the specified `<chat_id>`.
    *   Example: `/send 123456789 Hello from my bot!`
*   **Echo Functionality**: If you send any regular text message to the bot (not a command), it will echo that message back to you.
