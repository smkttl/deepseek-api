# DeepSeek API (unofficial)

This project reverse-engineered the Web Interface of DeepSeek at [it's official website](https://chat.deepseek.com) and provides simple script access to the otherwise complicated Web Interface.

## Features

- Simple API client for DeepSeek chat functionality.
- OpenAI-compatible HTTP server for easy integration with existing tools.
- Mostly bypasses the need for manual browser interaction.
- Lightweight and easy to integrate.
- Provides Markdown syntax for AI outputs.
- Supports both DeepSeek V3 (fast) and R1 (reasoning with extended thinking) models.

## Authentication

To use this API, you need two tokens from your DeepSeek account:

1. **ds_session_id** - A cookie value
2. **authorization_token** - A Bearer token in request headers

### How to obtain tokens

1. Log in to [chat.deepseek.com](https://chat.deepseek.com) in your browser
2. Open Developer Tools (F12) and go to the **Network** tab
3. Send any message in the chat
4. Find the POST request to `completion` in the network tab and click on it
5. In the request headers, find:
   - **Cookie** → Copy the `ds_session_id` value
   - **authorization** → Copy the Bearer token (e.g., `Bearer ...` (including the "Bearer " prefix))

### How to store tokens

Create a `tokens` file in the project root with two lines:

```
<ds_session_id>
<authorization_token>
```

Example `tokens` file:
```
35******************************775
Bearer sm*************************************************oT6
```

Alternatively, set environment variables:
```bash
export DS_SESSION_ID="your_session_id"
export AUTHORIZATION_TOKEN="Bearer your_token"
```

## Usage

### Option 1: CLI Script

Run the interactive CLI:
```bash
python main.py
```

### Option 2: OpenAI-Compatible Server

Start the HTTP server:
```bash
python server.py --host 0.0.0.0 --port 8000
```

Then use it with any OpenAI-compatible client:

```bash
# List available models
curl http://localhost:8000/v1/models

# Chat completion (non-streaming)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello!"}]}'

# Chat completion with DeepSeek R1 (reasoning model)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-r1", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Supported models:**
- `deepseek-chat` or `deepseek-v3` - Fast responses without extended thinking
- `deepseek-r1` or any model name containing "r1" or "reasoning" - Reasoning model with extended thinking

### Python API

```python
from DeepSeekAPI import DeepSeekChat, DeepSeekChatIOMethods

# Initialize with tokens
chat = DeepSeekChat(ds_session_id, authorization_token)

# Send message (returns dict with response)
result = chat.send_message("Hello!", thinking_enabled=False)

# For reasoning model
result = chat.send_message("Solve this problem", thinking_enabled=True)

# Stream output with markdown rendering
DeepSeekChatExample(
    ds_session_id,
    authorization_token,
    "Your message",
    DeepSeekChatIOMethods.STREAMDOWN,
    thinking_enabled=False,
    search_enabled=False
)
```

## To-do List

Future versions plan to include the following features:
- Web API usage
- A good wrapper and user-friendly interfaces.
- TBD

*You're welcome to submit your ideas and suggestions via Issues*

## Dependencies

This project depends on the wasmtime and streamdown libraries.

## Third-party assets

### streamdown
The `DeepSeekAPI/streamdown/` folder contains code from [Streamdown](https://github.com/day50-dev/Streamdown) and some of our modifications. All code in that directory, whether original or modified, remains under the [MIT license](DeepSeekAPI/streamdown/LICENSE.MIT).

**Files:**
- `DeepSeekAPI/plugins/latex.py` - Original Streamdown helper code for latex
- `DeepSeekAPI/streamdown/sd.py` - Original Streamdown code
- `DeepSeekAPI/streamdown/adapter.py` - Our adapter code
- `DeepSeekAPI/streamdown/__init__.py` - Our wrapper code

## Disclaimer

This project is **unofficial** and is not affiliated with, endorsed by, or connected to DeepSeek in any way. It works by reverse-engineering the DeepSeek web interface, which may violate DeepSeek's Terms of Service. 

**Use at your own risk.** The authors are not responsible for:

- Account bans or suspensions
- Data loss or privacy breaches
- Service disruptions
- Any other issues resulting from using this software

You are solely responsible for ensuring your use of this software complies with:
- DeepSeek's Terms of Service
- All applicable laws and regulations
- Your local jurisdiction's requirements

You agree not to use this software for any illegal, harmful, or abusive purposes.

## License

This project is licensed under the GNU General Affero Public License v3.0.

See the [LICENSE](LICENSE) file for the full text.

The streamdown module is licensed under the MIT License.

See the [LICENSE.MIT](DeepSeekAPI/streamdown/LICENSE.MIT) file for the full text.
