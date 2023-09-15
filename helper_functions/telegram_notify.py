# -*- coding: utf-8 -*-
"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: CogniGron
"""

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
        self.CHATIDs = asyncio.run(self._get_chatid()) if chat_id is None else chat_id

    def escape_markdown(self, text):
        for char in self.SPECIAL_CHARS:
            text = text.replace(char, '\\' + char)
        return text

    async def _get_chatid(self):
        chat_ids = []
        async with self.bot:
            try:
                updates = await self.bot.get_updates()
            except TimeoutError:
                return None
            try:
                for update in updates:
                    chat_ids.append(update.message.chat.id) if update.message.chat.id not in chat_ids else None
                return chat_ids
            except IndexError('No chat found, try messaging the bot first'):
                return None

    async def _send_message(self, text, chat_id):
        async with self.bot:
            msg = await self.bot.send_message(text=self.escape_markdown(text), chat_id=chat_id,
                                              parse_mode=telegram.constants.ParseMode.MARKDOWN_V2)
            return msg

    async def _edit_message(self, text, message_id, chat_id):
        async with self.bot:
            msg = await self.bot.edit_message_text(text=self.escape_markdown(text), chat_id=chat_id,
                                                   message_id=message_id,
                                                   parse_mode=telegram.constants.ParseMode.MARKDOWN_V2)
            return msg

    async def _reply_to_message(self, text, message_id, chat_id):
        async with self.bot:
            msg = await self.bot.send_message(text=self.escape_markdown(text), chat_id=chat_id,
                                              reply_to_message_id=message_id,
                                              parse_mode=telegram.constants.ParseMode.MARKDOWN_V2)
            return msg

    async def _append_to_message(self, text, message_id, chat_id):
        async with self.bot:
            msg = await self.bot.edit_message_text(text=self.escape_markdown(text), chat_id=chat_id,
                                                   message_id=message_id,
                                                   parse_mode=telegram.constants.ParseMode.MARKDOWN_V2)
            return msg

    async def _pin_message(self, message_id, chat_id):
        async with self.bot:
            msg = await self.bot.pin_chat_message(chat_id=chat_id, message_id=message_id)
            return msg

    async def _get_chat(self, chat_id):
        async with self.bot:
            chat = await self.bot.get_chat(chat_id=chat_id)
            return chat

    async def _unpin_all(self, chat_id):
        async with self.bot:
            await self.bot.unpin_all_chat_messages(chat_id=chat_id)

    def send_messages(self, text):
        messages = []
        for chat_id in self.CHATIDs:
            messages.append(
                asyncio.run(self._send_message(text, chat_id)))

        return messages

    def send_timestamped_messages(self, text):
        messages = []
        for chat_id in self.CHATIDs:
            messages.append(
                asyncio.run(self._send_message(f'*{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}* {text}', chat_id))
            )
        return messages

    def edit_messages(self, text, msgs):
        for m in msgs:
            asyncio.run(self._edit_message(text, m.id, m.chat_id))

    def edit_timestamped_messages(self, text, msgs):
        for m in msgs:
            asyncio.run(
                self._edit_message(f'*{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}* {text}', m.id, m.chat_id))

    def reply_to_message(self, text, msgs):
        for m in msgs:
            asyncio.run(self._reply_to_message(text, m.id, m.chat_id))

    def reply_to_timestamped_messages(self, text, msgs):
        messages = []
        for m in msgs:
            messages.append(
                asyncio.run(
                    self._reply_to_message(f'*{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}* {text}', m.id, m.chat_id))
            )
        return messages

    def pin_messages(self, msgs):
        for m in msgs:
            asyncio.run(self._pin_message(m.id, m.chat_id))

    def get_chats(self):
        chats = []
        for chat_id in self.CHATIDs:
            chats.append(asyncio.run(self._get_chat(chat_id)))
        return chats

    def unpin_all(self):
        for chat_id in self.CHATIDs:
            asyncio.run(self._unpin_all(chat_id))

    def read_pinned_and_increment_it(self, msgs, score):
        import re

        last_updates = []

        self.pin_messages(msgs)
        chats = self.get_chats()
        for chat in chats:
            try:
                last_updates.append((chat.id,
                                     int(re.findall(r"\d+", chat.pinned_message.text_markdown_v2)[6]),
                                     int(re.findall(r"\d+", chat.pinned_message.text_markdown_v2)[7]),
                                     ))
                self.unpin_all()
                max_up = max(last_updates, key=lambda x: x[1])
                self.edit_timestamped_messages(f'*{max_up[1] + 1}/{max_up[2]}* Finished run. Score: {score}', msgs)
            except:
                pass
