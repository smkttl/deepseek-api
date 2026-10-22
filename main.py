#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DeepSeek API - Provides an unofficial API for DeepSeek by reverse-engineering its web interface.
Copyright (C) 2025 smkttl

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from DeepSeekAPI import *
if __name__ == "__main__":
    with open("tokens") as f:
        DS_SESSION_ID,AUTHORIZATION_TOKEN = f.read().strip().split('\n')
    print(DeepSeekChatExample(DS_SESSION_ID,AUTHORIZATION_TOKEN,input('Your message:'),DeepSeekChatIOMethods.STREAMDOWN,False,False))
