# API Reference

```python
from supertonic import TTS

tts = TTS(auto_download=True)       # Initialize TTS engine

style = tts.get_voice_style(voice_name="M1")   # 10 built-in voices: M1–M5, F1–F5

wav, duration = tts.synthesize(
    text="Your text here.",         # Text to synthesize
    voice_style=style,              # Voice style object
    total_steps=8,                  # Quality: 5 (low) to 12 (high), default 8
    speed=1.05,                     # Speed: 0.7 (slow) to 2.0 (fast)
    max_chunk_length=300,           # Max characters per chunk (auto: 120 for Korean)
    silence_duration=0.3,           # Silence between chunks (seconds)
    lang=None,                      # Auto: "na" for multilingual, "en" for v1
    verbose=False,                  # Show detailed progress (default: False)
)
```

## Modules

- **[pipeline](pipeline.md)** - High-level TTS interface with automatic model loading and voice style management
- **[streaming](streaming.md)** - Incremental, clause-by-clause synthesis for realtime voice agents
- **[core](core.md)** - Core TTS engine classes and data structures
- **[loader](loader.md)** - Functions for loading models and voice styles
- **[utils](utils.md)** - Helper functions for text processing and audio utilities
- **[config](config.md)** - Configuration constants and default values
- **[cli](cli.md)** - Command-line interface implementation
- **[server](server.md)** - Optional FastAPI HTTP server (`pip install supertonic[serve]`)
