# Morpheon 3

A text-to-video generation Cog powered by Ovi.

## Overview

Morpheon 3 is a Replicate-deployable model that generates videos from text descriptions using the Ovi pipeline.

## Usage

### Local Testing

```bash
# Install Cog
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
chmod +x /usr/local/bin/cog

# Run a prediction locally
cog predict -i prompt="A cat walking through a garden"
```

### Deploy to Replicate

```bash
# Login to Replicate
cog login

# Push to Replicate
cog push r8.im/<your-username>/morpheon3
```

## Inputs

- **prompt**: Text description of the video to generate
- **negative_prompt**: Things to avoid in the generated video
- **num_frames**: Number of frames (8-64, default: 16)
- **num_inference_steps**: Denoising steps (10-100, default: 25)
- **guidance_scale**: CFG scale (1.0-20.0, default: 7.5)
- **fps**: Output video frame rate (4-30, default: 8)
- **width**: Video width (128-1024, default: 256)
- **height**: Video height (128-1024, default: 256)
- **seed**: Random seed for reproducibility

## Output

Returns an MP4 video file generated from the text prompt.

## License

MIT
