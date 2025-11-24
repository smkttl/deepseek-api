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

import requests
import json
import time
import random
from traceback import format_exc as backtrace
class DeepSeekChat:
    def __init__(self,ds_session_id,authorization_token):
        self.session=requests.Session()
        self.base_url="https://chat.deepseek.com"
        self.session.cookies.set("ds_session_id",ds_session_id)
        self.authorization=authorization_token
        self.headers={
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,fr;q=0.8",
            "authorization": self.authorization,
            "content-type": "application/json",
            "priority": "u=1, i",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-app-version": "20241129.1",
            "x-client-locale": "zh_CN",
            "x-client-platform": "web",
            "x-client-version": "1.5.0",
            "x-debug-lite-model-channel": "prod",
            "x-debug-model-channel": "prod",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/142.0"
        }
        self.chat_session_id = None
        self.parent_message_id = None
    def create_chat_session(self):
        url=f"{self.base_url}/api/v0/chat_session/create"
        response = self.session.post(url,headers=self.headers,data="{}",timeout=30)
        if response.status_code==200:
            result = response.json()
            if result.get("code")==0:
                self.chat_session_id=result["data"]["biz_data"]["id"]
                return True
            else:
                return False
        else:
            return False
    def create_pow_challenge(self):
        url=f"{self.base_url}/api/v0/chat/create_pow_challenge"
        data={"target_path":"/api/v0/chat/completion"}
        headers=self.headers.copy()
        if self.chat_session_id:
            headers["referrer"]=f"{self.base_url}/a/chat/s/{self.chat_session_id}"
        response=self.session.post(url,headers=headers,data=json.dumps(data),timeout=30)
        if response.status_code == 200:
            result=response.json()
            if result.get("code") == 0:
                challenge_data=result["data"]["biz_data"]["challenge"]
                return challenge_data
            else:
                return None
        else:
            return None
    def solve_pow_challenge(self, challenge_data):
        try:
            from .DeepSeekWASM import solve_wasm
            algorithm = challenge_data["algorithm"]
            challenge = challenge_data["challenge"]
            salt = challenge_data["salt"]
            expire_at = challenge_data["expire_at"]
            difficulty = challenge_data["difficulty"]
            signature = challenge_data["signature"]
            target_path = challenge_data["target_path"]
            value, pow_response = solve_wasm(algorithm, challenge, salt, expire_at, difficulty, signature, target_path)
            if value:
                return True, pow_response
            return False,''
        except ImportError:
            raise
        except Exception as e:
            return False,''
    def generate_client_stream_id(self):
        timestamp = time.strftime("%Y%m%d")
        random_hex = ''.join(random.choices('0123456789abcdef', k=16))
        return f"{timestamp}-{random_hex}"
    def send_message(self,message,printing=None,thinking_enabled=False,search_enabled=False):
        if not self.chat_session_id:
            if not self.create_chat_session():
                return {"ok": False, "content": "Can't create chat session."}
        challenge_data = self.create_pow_challenge()
        if not challenge_data:
            return {"ok": False, "content": "Can't create PoW challenge."}
        value,pow_response = self.solve_pow_challenge(challenge_data)
        if not value:
            return {"ok": False, "content": "Can't solve PoW challenge."}
        url = f"{self.base_url}/api/v0/chat/completion"
        headers = self.headers.copy()
        headers["x-ds-pow-response"] = pow_response
        headers["accept"] = "text/event-stream"
        if self.chat_session_id:
            headers["referrer"] = f"{self.base_url}/a/chat/s/{self.chat_session_id}"
        data = {
            "chat_session_id": self.chat_session_id,
            "parent_message_id": self.parent_message_id,
            "prompt": message,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "client_stream_id": self.generate_client_stream_id(),
            "stream": True
        }
        try:
            response = self.session.post(
                url,
                headers=headers,
                data=json.dumps(data),
                timeout=60,
                stream=True
            )
            if response.status_code == 200:
                content_type=response.headers.get('content-type', '')
                if not "text/event-stream" in content_type:
                    full_content=b''
                    for chunk in response.iter_content(chunk_size=8192):
                        full_content += chunk
                    return {"ok": False, "content": full_content}
                event='UNKNOWN'#ready=preparing update_session=outputting title close
                think=''
                respond=''
                generate_mode=''#THINK RESPONSE SEARCH TIP(This content is AI-generated... stuff)
                parid,msgid,reqid,resid=None,None,None,None #suspect parid=reqid msgid=resid NOT SURE
                tokencount=None
                title=''
                thinktime=0
                citation={}
                sd_remains=""
                def send_to_sd(text):
                    print(text,end='',flush=True)
                def parse_output(data,line):
                    nonlocal event,think,respond,generate_mode,parid,msgid,thinktime
                    if not data:
                        return
                    if type(data)==bool:
                        return
                    if type(data)==str:
                        if generate_mode=='THINK':
                            think+=data
                        elif generate_mode=='RESPONSE':
                            respond+=data
                        elif generate_mode=='TIP':
                            pass
                        else:
                            raise Exception(f"Unexptected string in mode {generate_mode}\nData: {line}")
                        if printing:
                            send_to_sd(data)
                        return
                    if type(data)==list:
                        for d in data:
                            parse_output(d,line)
                        return
                    if type(data)==dict:
                        if generate_mode=='SEARCH' and 'url' in data:
                            citation[data.get('cite_index')]=data
                            if printing:
                                send_to_sd(f"{data.get('cite_index','?')}. [{data.get('title',data.get('site_name','UNKNOWN'))} - {data.get('site_name','UNKNOWN')}]({data['url']})\n")
                                send_to_sd('> '+data.get('snippet','')+"\n")
                        elif 'v' in data and len(data)==1:
                            parse_output(data['v'],line)
                        elif 'response' in data and len(data)==1:
                            parse_output(data['response'],line)
                        elif 'message_id' in data or 'parent_id' in data:
                            parid=data.get('parent_id',parid)
                            msgid=data.get('message_id',msgid)
                        elif 'updated_at' in data and len(data)==1:
                            return #CURRENTLY no use of this value
                        elif 'p' in data:
                            tp=data['p'].split('/')[-1]
                            if tp in ['response','content','fragment','fragments']:
                                parse_output(data['v'],line)
                            elif tp=='status':
                                if printing:
                                    send_to_sd('\n\n-----\n'+data['v'])
                            elif tp=='accumulated_token_usage':
                                tokencount=data['v']
                            elif tp=='elapsed_secs':
                                thinktime=data['v']
                            elif tp=='results' and generate_mode=='SEARCH':
                                parse_output(data['v'],line)
                            elif tp=='has_pending_fragment' or tp=='conversation_mode':
                                pass
                            else:
                                raise Exception(f"Unknown 'p' field: {tp}\nData: {line}")
                        elif 'type' in data and data['type']:
                            if generate_mode!=data['type']:
                                if printing:
                                    send_to_sd(f"\n\n-----\nSTART {data['type']}\n")
                                generate_mode=data['type']
                            if 'content' in data:
                                parse_output(data['content'],line)
                        else:
                            raise Exception(f"Cannot parse dict {json.dumps(data,separators=(',',':'))}\nData: {line}")
                    else:
                        raise Exception(f"Unrecognizable type {type(data)}\nData: {line}")
                send_to_sd('\n')
                for line in response.iter_lines(decode_unicode=True):
                    if line and len(line)>0:
                        if type(line)==str:
                            if line.startswith('data: '):
                                data=json.loads(line[6:])
                                if event=='update_session':
                                    parse_output(data,line)
                                elif event in ['finish','close']:
                                    pass
                                elif event=='title' and 'content' in data:
                                    title=data['content']
                                elif event=='ready' and ('request_message_id' in data or 'response_message_id' in data):
                                    reqid=data.get('request_message_id',reqid)
                                    resid=data.get('response_message_id',resid)
                                else:
                                    raise Exception(f"Unknown event: {event}\nData: {line}")
                            elif line.startswith('event: '):
                                event=line[7:]
                            else:
                                raise Exception(f"Unrecognizable line: {line}\nData: {line}")
                        else:
                            raise Exception(f"Unexpected response consisting of {type(line)}")
                if printing:
                    print(f"\nFinished generating... Thinking time: {thinktime} Total tokens: {tokencount} {'Title: '+title if len(title)>0 else ''}")
                ret={}
                if thinking_enabled:
                    ret["thinktime"]=thinktime
                    ret["thought"]=think
                if search_enabled:
                    ret["citation"]=citation
                ret["thinking_enabled"]=thinking_enabled
                ret["search_enabled"]=search_enabled
                ret["response"]=respond
                if len(title)>0:
                    ret["title"]=title
                return {"ok": True, "content": ret}
            else:
                print(f"HTTP ERROR: {response.status_code}\n{response.text}")
                return None
        except Exception as e:
            print(backtrace())
            return {"ok": False, "content": str(e)}
