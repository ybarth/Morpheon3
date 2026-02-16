"""
Morpheon 3 - Ovi Text-to-Video Predictor
A Cog implementation for text-to-video generation using Ovi.
"""

import os
import tempfile
from typing import Optional

import torch
from cog import BasePredictor, Input, Path


class OviPredictor(BasePredictor):
    """Ovi text-to-video generation predictor."""

    def setup(self) -> None:
        """Load the Ovi model into memory for efficient inference."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Initialize the Ovi text-to-video pipeline
        # TODO: Replace with actual Ovi model loading
        from diffusers import DiffusionPipeline
        
        self.pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=self.dtype,
            variant="fp16" if self.device == "cuda" else None,
        )
        self.pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()

    def predict(
        self,
        prompt: str = Input(
            description="Text description of the video to generate",
            default="A serene ocean wave rolling onto a sandy beach at sunset"
        ),
        negative_prompt: str = Input(
            description="Things to avoid in the generated video",
            default="low quality, blurry, distorted, ugly, bad anatomy"
        ),
        num_frames: int = Input(
            description="Number of frames to generate",
            default=16,
            ge=8,
            le=64
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=25,
            ge=10,
            le=100
        ),
        guidance_scale: float = Input(
            description="Guidance scale for classifier-free guidance",
            default=7.5,
            ge=1.0,
            le=20.0
        ),
        fps: int = Input(
            description="Frames per second for output video",
            default=8,
            ge=4,
            le=30
        ),
        width: int = Input(
            description="Width of generated video",
            default=256,
            ge=128,
            le=1024
        ),
        height: int = Input(
            description="Height of generated video",
            default=256,
            ge=128,
            le=1024
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducibility (leave empty for random)",
            default=None
        ),
    ) -> Path:
        """Generate a video from text description using Ovi."""
        import imageio
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate video frames
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )

        # Extract frames
        frames = output.frames[0]

        # Save as MP4
        output_path = Path(tempfile.mktemp(suffix=".mp4"))
        imageio.mimsave(
            str(output_path),
            frames,
            fps=fps,
            codec="libx264",
            quality=8,
        )

        return output_path
