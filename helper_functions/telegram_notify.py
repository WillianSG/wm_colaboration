import asyncio
import telegram
from datetime import datetime


class TelegramNotify:

    def __init__(self, token=None, chat_id=None):
        self.SPECIAL_CHARS = [
            '\\',
            '_',
            '[',
            ']',
            '(',
            ')',
            '~',
            '`',
            '>',
            '<',
            '&',
            '#',
            '+',
            '-',
            '=',
            '|',
            '{',
            '}',
            '.',
            '!'
        ]

        self.TOKEN = '6491481149:AAFomgrhyBRohH4szH5jPT2_AoAdOYA_flY' if token is None else token

        self.bot = telegram.Bot(self.TOKEN)
        self.CHATID = asyncio.run(self._get_chatid()) if chat_id is None else chat_id

    def escape_markdown(self, text):
        for char in self.SPECIAL_CHARS:
            text = text.replace(char, '\\' + char)
        return text

    async def _get_chatid(self):
        async with self.bot:
            updates = await self.bot.get_updates()
            return updates[-1].message.chat.id

    async def _send_message(self, text):
        async with self.bot:
            msg = await self.bot.send_message(text=self.escape_markdown(text), chat_id=self.CHATID,
                                              parse_mode=telegram.constants.ParseMode.MARKDOWN_V2)
            return msg.message_id

    async def _edit_message(self, text, message_id):
        async with self.bot:
            msg = await self.bot.edit_message_text(text=self.escape_markdown(text), chat_id=self.CHATID,
                                                   message_id=message_id,
                                                   parse_mode=telegram.constants.ParseMode.MARKDOWN_V2)
            return msg.message_id

    async def _reply_to_message(self, text, message_id):
        async with self.bot:
            msg = await self.bot.send_message(text=self.escape_markdown(text), chat_id=self.CHATID,
                                              reply_to_message_id=message_id,
                                              parse_mode=telegram.constants.ParseMode.MARKDOWN_V2)
            return msg.message_id

    async def _append_to_message(self, text, message_id):
        async with self.bot:
            msg = await self.bot.edit_message_text(text=self.escape_markdown(text), chat_id=self.CHATID,
                                                   message_id=message_id,
                                                   parse_mode=telegram.constants.ParseMode.MARKDOWN_V2)
            return msg.message_id

    def send_message(self, text):
        return asyncio.run(self._send_message(text))

    def send_timestamped_message(self, text):
        return asyncio.run(self._send_message(f'*{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}* {text}'))

    def edit_message(self, text, message_id):
        return asyncio.run(self._edit_message(text, message_id))

    def edit_timestamped_message(self, text, message_id):
        return asyncio.run(self._edit_message(f'*{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}* {text}', message_id))

    def reply_to_message(self, text, message_id):
        return asyncio.run(self._reply_to_message(text, message_id))

    def reply_to_timestamped_message(self, text, message_id):
        return asyncio.run(
            self._reply_to_message(f'*{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}* {text}', message_id))
