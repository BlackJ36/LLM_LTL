"""
Qwen3-VL API Client for VLM-guided RL reward.

Communicates with a locally deployed Qwen3-VL-8B-Instruct model
via OpenAI-compatible API.
"""

import base64
import json
import re
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import requests
from PIL import Image


class QwenVLClient:
    """Client for Qwen3-VL model deployed with OpenAI-compatible API."""

    def __init__(
        self,
        api_base: str = "http://172.19.1.40:8001",
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        timeout: float = 30.0,
    ):
        """
        Initialize the VLM client.

        Args:
            api_base: Base URL of the VLM API server
            model_name: Model identifier for the API
            timeout: Request timeout in seconds
        """
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.endpoint = f"{self.api_base}/v1/chat/completions"

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _encode_images(self, images: list[np.ndarray]) -> list[str]:
        """Convert a list of numpy array images to base64 strings."""
        return [self._encode_image(img) for img in images]

    # Skill descriptions for prompt
    SKILL_DESCRIPTIONS = {
        "atomic": "Single-step low-level action. Directly applies small position delta to the end-effector. Use for fine adjustments or when other skills are not appropriate.",
        "reach": "Multi-step reaching skill. Moves the gripper to a target 3D position using a safe trajectory: first lifts up, then moves horizontally, then descends. Gripper state can be controlled. Use when you need to position the gripper near an object.",
        "grasp": "Multi-step grasping skill. Reaches a target position and closes the gripper to grasp an object. Opens gripper while approaching, closes when reached. Use when the gripper is near an object and ready to pick it up.",
        "push": "Multi-step pushing skill. Reaches a source position then pushes toward a target position. Gripper stays closed. Use to slide or push objects on the surface.",
        "reach_osc": "OSC-based reaching skill. Similar to reach but uses operational space control for smoother motion. Use for precise positioning tasks.",
        "open": "Opens the gripper. Use after placing an object or to release a grasped item.",
        "close": "Closes the gripper. Use to grasp an object when positioned correctly.",
    }

    def _get_skill_descriptions(self, available_primitives: list[str]) -> str:
        """Generate skill descriptions for the available primitives."""
        descriptions = []
        for skill in available_primitives:
            desc = self.SKILL_DESCRIPTIONS.get(skill, f"Unknown skill: {skill}")
            descriptions.append(f"  - {skill}: {desc}")
        return "\n".join(descriptions)

    def _build_messages(
        self,
        image: np.ndarray,
        task_description: str,
        selected_primitive: str,
        available_primitives: list[str],
        evaluation_type: str = "binary",
        image_history: list[np.ndarray] = None,
    ) -> list[dict]:
        """Build the message payload for the API request.

        Args:
            image: Current scene image
            task_description: Task description
            selected_primitive: Selected action primitive
            available_primitives: List of available primitives
            evaluation_type: "binary" or "progress"
            image_history: Optional list of recent images (oldest to newest)
        """
        skill_descriptions = self._get_skill_descriptions(available_primitives)

        # Build image content list
        image_content = []

        if image_history and len(image_history) > 0:
            # Add historical images with labels
            num_history = len(image_history)
            for i, hist_img in enumerate(image_history):
                img_b64 = self._encode_image(hist_img)
                # Add image
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })
                # Add label
                image_content.append({
                    "type": "text",
                    "text": f"[Frame {i+1}/{num_history+1}: {num_history-i} steps ago]"
                })

            # Add current image
            current_b64 = self._encode_image(image)
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{current_b64}"}
            })
            image_content.append({
                "type": "text",
                "text": f"[Frame {num_history+1}/{num_history+1}: CURRENT - action being evaluated]"
            })
        else:
            # Single image mode
            image_b64 = self._encode_image(image)
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            })

        if evaluation_type == "binary":
            if image_history and len(image_history) > 0:
                sequence_note = f"""
The images above show a sequence of {len(image_history)+1} frames (oldest to newest).
Observe how the robot has been moving and use this context to evaluate the action selection.
The CURRENT frame shows the state when the action "{selected_primitive}" was selected."""
            else:
                sequence_note = ""

            prompt = f"""You are evaluating a robot's action selection for a manipulation task.

Task: {task_description}

Available action primitives and their functions:
{skill_descriptions}

The robot selected: "{selected_primitive}"
{sequence_note}

Based on the scene(s) shown:
1. Observe the robot gripper position relative to the target object
2. Consider the motion trajectory if multiple frames are provided
3. Evaluate if the selected primitive is appropriate for the current state

Key considerations (adapt based on task type):

For MANIPULATION tasks (lifting, stacking, pick-place):
- If gripper is far from object, "reach" is appropriate
- If gripper is near/above object, "grasp" is appropriate
- If object needs to be moved on surface, "push" is appropriate

For DOOR tasks (opening doors):
- If gripper is far from door handle, "reach" is appropriate
- If gripper is near handle, "grasp" to grip the handle
- Once gripping, use repeated "atomic" actions to push/pull the door - this is EXPECTED behavior
- "open" to release gripper if needed
- NOTE: Using "atomic" repeatedly while manipulating the door is REASONABLE

For all tasks:
- "atomic" is for fine-grained position adjustments
- Consider the motion trajectory shown in the image sequence

Respond with ONLY a JSON object in this exact format:
{{"reasonable": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        elif evaluation_type == "progress":
            if image_history and len(image_history) > 0:
                sequence_note = f"""
The images above show a sequence of {len(image_history)+1} frames (oldest to newest).
Compare the frames to assess how much progress has been made."""
            else:
                sequence_note = ""

            prompt = f"""You are evaluating task progress for a robot manipulation task.

Task: {task_description}
{sequence_note}

Based on the scene(s) shown, analyze:
1. Current gripper position and state (open/closed)
2. Object position relative to the goal
3. Motion trend if multiple frames are provided
4. Overall progress toward task completion

Progress stages for reference:
- 0.0-0.2: Initial state, gripper far from object
- 0.2-0.4: Gripper approaching the object
- 0.4-0.6: Gripper positioned near object, ready to interact
- 0.6-0.8: Object grasped or being manipulated
- 0.8-1.0: Object near goal position or task nearly complete

Respond with ONLY a JSON object in this exact format:
{{"progress": 0.0-1.0, "stage": "description of current stage", "next_step": "what should happen next"}}"""

        else:
            raise ValueError(f"Unknown evaluation_type: {evaluation_type}")

        # Combine image content with prompt
        content = image_content + [{"type": "text", "text": prompt}]

        messages = [{"role": "user", "content": content}]
        return messages

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from model response, handling potential formatting issues."""
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Return default if parsing fails
        return {"error": "Failed to parse response", "raw": text}

    def evaluate_action(
        self,
        image: np.ndarray,
        task_description: str,
        selected_primitive: str,
        available_primitives: list[str],
        image_history: list[np.ndarray] = None,
    ) -> Tuple[bool, float, str]:
        """
        Evaluate if the selected action primitive is reasonable.

        Args:
            image: RGB image of the current scene (H, W, 3)
            task_description: Natural language description of the task
            selected_primitive: Name of the selected action primitive
            available_primitives: List of all available primitive names
            image_history: Optional list of recent images for context (oldest to newest)

        Returns:
            Tuple of (is_reasonable, confidence, reason)
        """
        messages = self._build_messages(
            image=image,
            task_description=task_description,
            selected_primitive=selected_primitive,
            available_primitives=available_primitives,
            evaluation_type="binary",
            image_history=image_history,
        )

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            parsed = self._parse_json_response(content)

            return (
                parsed.get("reasonable", False),
                parsed.get("confidence", 0.5),
                parsed.get("reason", "No reason provided"),
            )

        except Exception as e:
            # On failure, return neutral values to avoid disrupting training
            return (True, 0.5, f"VLM error: {str(e)}")

    def evaluate_progress(
        self,
        image: np.ndarray,
        task_description: str,
        image_history: list[np.ndarray] = None,
    ) -> Tuple[float, str, str]:
        """
        Evaluate task progress from the current scene.

        Args:
            image: RGB image of the current scene (H, W, 3)
            task_description: Natural language description of the task
            image_history: Optional list of recent images for context (oldest to newest)

        Returns:
            Tuple of (progress 0-1, stage_description, next_step)
        """
        messages = self._build_messages(
            image=image,
            task_description=task_description,
            selected_primitive="",
            available_primitives=[],
            evaluation_type="progress",
            image_history=image_history,
        )

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            parsed = self._parse_json_response(content)

            return (
                float(parsed.get("progress", 0.0)),
                parsed.get("stage", "Unknown"),
                parsed.get("next_step", "Unknown"),
            )

        except Exception as e:
            return (0.0, "Error", f"VLM error: {str(e)}")

    def get_combined_reward(
        self,
        image: np.ndarray,
        task_description: str,
        selected_primitive: str,
        available_primitives: list[str],
        prev_progress: Optional[float] = None,
        binary_weight: float = 0.5,
        progress_weight: float = 0.5,
        reward_mode: str = "bonus_only",
        image_history: list[np.ndarray] = None,
    ) -> Tuple[float, dict]:
        """
        Get combined VLM reward from binary judgment and progress evaluation.

        Args:
            image: RGB image of the current scene
            task_description: Natural language task description
            selected_primitive: Selected action primitive name
            available_primitives: All available primitive names
            prev_progress: Previous progress value for delta calculation
            binary_weight: Weight for binary action evaluation reward
            progress_weight: Weight for progress-based reward
            reward_mode: How to compute reward:
                - "bonus_only": Only positive rewards (no penalty for bad choices)
                - "symmetric": Both positive and negative rewards
                - "penalty_only": Only negative rewards (penalize bad choices)
            image_history: Optional list of recent images for context (oldest to newest)

        Returns:
            Tuple of (total_reward, info_dict)
        """
        # Binary action evaluation
        is_reasonable, confidence, reason = self.evaluate_action(
            image=image,
            task_description=task_description,
            selected_primitive=selected_primitive,
            available_primitives=available_primitives,
            image_history=image_history,
        )

        # Compute binary reward based on mode
        if reward_mode == "bonus_only":
            # Only reward good choices, don't penalize bad ones
            binary_reward = confidence if is_reasonable else 0.0
        elif reward_mode == "penalty_only":
            # Only penalize bad choices, don't reward good ones
            binary_reward = -confidence if not is_reasonable else 0.0
        else:  # symmetric
            binary_reward = confidence if is_reasonable else -confidence

        # Progress evaluation
        progress, stage, next_step = self.evaluate_progress(
            image=image,
            task_description=task_description,
            image_history=image_history,
        )

        # Progress reward: delta-based to avoid reward hacking
        if prev_progress is not None:
            progress_delta = progress - prev_progress
            # Only reward progress improvement, don't penalize regression
            # (regression is already penalized by env reward)
            if reward_mode == "bonus_only":
                progress_reward = max(0, progress_delta)
            elif reward_mode == "penalty_only":
                progress_reward = min(0, progress_delta)
            else:
                progress_reward = progress_delta
        else:
            progress_reward = 0.0  # No reward on first step

        # Combined reward
        total_reward = binary_weight * binary_reward + progress_weight * progress_reward

        info = {
            "vlm_binary_reasonable": is_reasonable,
            "vlm_binary_confidence": confidence,
            "vlm_binary_reason": reason,
            "vlm_progress": progress,
            "vlm_stage": stage,
            "vlm_next_step": next_step,
            "vlm_binary_reward": binary_reward,
            "vlm_progress_reward": progress_reward,
            "vlm_total_reward": total_reward,
            "vlm_reward_mode": reward_mode,
        }

        return total_reward, info

    def score_all_primitives(
        self,
        image: np.ndarray,
        task_description: str,
        available_primitives: list[str],
        image_history: list[np.ndarray] = None,
    ) -> dict[str, float]:
        """
        Score all available primitives for the current state (Plan C: Soft Scoring).

        Instead of binary judgment, this returns a suitability score (0-1) for each
        primitive, allowing for softer reward signals.

        Args:
            image: RGB image of the current scene (H, W, 3)
            task_description: Natural language description of the task
            available_primitives: List of all available primitive names
            image_history: Optional list of recent images for context (oldest to newest)

        Returns:
            Dict mapping primitive name to suitability score (0.0-1.0)
            Example: {"atomic": 0.3, "reach": 0.8, "grasp": 0.1, "push": 0.2}
        """
        skill_descriptions = self._get_skill_descriptions(available_primitives)

        # Build image content
        image_content = []

        if image_history and len(image_history) > 0:
            num_history = len(image_history)
            for i, hist_img in enumerate(image_history):
                img_b64 = self._encode_image(hist_img)
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })
                image_content.append({
                    "type": "text",
                    "text": f"[Frame {i+1}/{num_history+1}: {num_history-i} steps ago]"
                })

            current_b64 = self._encode_image(image)
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{current_b64}"}
            })
            image_content.append({
                "type": "text",
                "text": f"[Frame {num_history+1}/{num_history+1}: CURRENT]"
            })
        else:
            image_b64 = self._encode_image(image)
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            })

        # Build expected output format for prompt
        example_output = {p: 50 for p in available_primitives}

        prompt = f"""You are evaluating action primitives for a robot manipulation task.

Task: {task_description}

Available action primitives and their functions:
{skill_descriptions}

For each primitive, rate how appropriate it would be to execute in the current state.
Score from 0 (completely inappropriate) to 100 (ideal action).

Consider:
- Current gripper position relative to objects
- Current gripper state (open/closed)
- What step the task is at
- What action would make the most progress

Respond with ONLY a JSON object mapping each primitive to its score (0-100):
{json.dumps(example_output)}"""

        content = image_content + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            parsed = self._parse_json_response(content)

            # Convert scores from 0-100 to 0-1 and ensure all primitives have scores
            scores = {}
            for primitive in available_primitives:
                raw_score = parsed.get(primitive, 50)  # Default to 50 if missing
                # Handle both int and string values
                if isinstance(raw_score, str):
                    try:
                        raw_score = float(raw_score)
                    except ValueError:
                        raw_score = 50
                scores[primitive] = max(0.0, min(1.0, raw_score / 100.0))

            return scores

        except Exception as e:
            # On failure, return neutral scores for all primitives
            return {p: 0.5 for p in available_primitives}
