# DeepSeek API (unofficial)

This project reverse-engineered the Web Interface of DeepSeek at [it's official website](https://chat.deepseek.com) and provides simple script access to the otherwise complicated Web Interface.

## Features

- Simple API client for DeepSeek chat functionality.
- Mostly bypasses the need for manual browser interaction.
- Lightweight and easy to integrate.
- Provides Markdown syntax for AI outputs.

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
