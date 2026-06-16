#!/usr/bin/env python
# coding: utf-8

# # Run NaviTrace Evaluation
# 
# This notebook describes the process of evaluating models on our benchmark [NaviTrace](https://leggedrobotics.github.io/navitrace_webpage/), including model inference via API and the score calculation.
# The benchmark consists of a validation split and a test split with hidden ground-truths.
# If you want to see how your model scores on the test set or want to submit your model to the leaderboard, check out this [Hugging Face Space](https://huggingface.co/spaces/leggedrobotics/navitrace_leaderboard).
# 
# ## Setup
# 
# 1. Create and activate a Python 3.10 environment with your preferred tools
# 2. `pip install -r ./requirements.txt`
# 3. Prepare an API key and base URL for the model that you want to evaluate

# ## Load NaviTrace

# In[1]:


from datasets import load_dataset
from itables import show


# In[2]:


# (Optional) Login at HF
from huggingface_hub import login
login(token="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")


# In[3]:


# Load dataset
dataset = load_dataset("leggedrobotics/navitrace")


# Have a look at the [dataset card](https://huggingface.co/datasets/leggedrobotics/navitrace) for information about the available columns. You can also explore the dataset with the following code:

# In[4]:



# In[6]:


import base64
import io
import json
import re
import time
from typing import Dict, List, Any
from datetime import datetime
from getpass import getpass
import os
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import pandas as pd
from PIL.Image import Image
from tqdm import tqdm

# Set environment variable for vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


# ### Setup Model (vLLM Offline)

# In[7]:


# Settings
# Note: Using vLLM offline inference instead of API
model_name = os.environ.get("MODEL_PATH")
#model_name = "/path/to/Quadruped_Reasoning/Qwen3-VL/qwen-vl-finetune/checkpoints/checkpointsq3_4b_disjoint_images_20drop_hugedata_it1/checkpoint-404"
base_url = "http://127.0.0.1:8000/v1"  # Not used for offline inference, kept for compatibility
model_safe_name = os.environ.get("EXPERIMENT_NAME")
if model_safe_name is None:
    model_safe_name = model_name.replace("/", "_").replace(":", "_")
# ### Define Prompts

# In[8]:


system_prompt = """\nYou are a navigation expert for various embodiments including robots and humans. Given an image of the current scenario, a specified embodiment (e.g., legged robot, wheeled robot, human, or bike), and a navigation task (e.g., "Go down the road"), you will predict a feasible future trajectory as a sequence of 2D points in normalized image coordinates (ranging from 0 to 1, where [0,0] is the top-left and [1,1] is the bottom-right).
  - The image shows a first-person view of the navigation scenario
  - Start your trajectory near the bottom center of the image, which corresponds approximately to normalized coordinate [0.5, 0.95] (representing the current position of the embodiment)
  - The trajectory should be adapted to the embodiment's abilities and limitations
  - Plan the path forward from this starting position based on what the embodiment can see and navigate
  - The trajectory should extend all the way to the goal if the path is visible. If the path is occluded, the trajectory should end where the path becomes fully obscured, unless the path can be reasonably inferred from the visible context.
  - All tasks that you are given have a solution
  - Output **only** the list of 2D points in normalized image coordinates (values between 0 and 1) in the following format: `[[x1, y1], [x2, y2], ..., [xn, yn]]`
  - Do not include any explanation or additional output
  ### Embodiment Movement Characteristics
  - **Human**: A standard pedestrian. Can navigate stairs and ramps but cannot climb tall obstacles.
  - **Legged Robot**: A quadruped like ANYmal. Behaves similarly to a human, but it is shorter. It can handle stairs and escalators.
  - **Wheeled Robot**: A wheeled delivery robot. Behaves like a wheelchair, preferring smooth surfaces such as walkways and ramps. It cannot use stairs or escalators.
  - **Bicycle**: A standard cyclist. Follows traffic regulations and prefers bike lanes or streets. Cannot navigate stairs.
"""

system_prompt += """**Embodiment**: {embodiment}\n**Task**: {task_str}\n"""



# ### Define vLLM Model Class
# 
# Processing a single sample produces a dict with the form:
# 
# | Column | Type | Description |
# | --- | ---- | ----------- |
# | sample_id | `str` | Unique identifier of a scenario |
# | embodiment | `str` | Selected embodiment |
# | category | `List[str]` | Scenario categories |
# | raw_response | `str` | Raw text response of the model |
# | reasoning | `str` | If available, the reasoning output of the model |
# | prediction | `List[List[float]]` | List of [x, y] points representing the predicted trace |

# In[9]:


def encode_image_to_base64(image: Image) -> str:

    # Convert to RGB if necessary
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    # Save image to a bytes buffer as JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    # Encode buffer in base64
    img_bytes = buffer.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return img_b64

def prepare_inputs_for_vllm(messages, processor):
    """Prepare inputs for vLLM inference using the same format as rnbencore_vllm_offline.py"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def parse_trace(text: str) -> List[List[float]]:
    """
    Parse point trace from model response.
    Expected format: [[x1, y1], [x2, y2], ...] or similar variations.
    Returns a list of [x, y] coordinate pairs, or an empty list if parsing fails.
    """

    try:
        # Try to find JSON-like array in response
        patterns = [
            r"\[\s*\[[\d\s,.-]+\]\s*(?:,\s*\[[\d\s,.-]+\]\s*)*\]",  # [[x,y], [x,y], ...]
            r"\(\s*\([\d\s,.-]+\)\s*(?:,\s*\([\d\s,.-]+\)\s*)*\)",  # ((x,y), (x,y), ...)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Parse the first match
                match = matches[0]
                # Convert to proper JSON format
                match = match.replace("(", "[").replace(")", "]")
                points = json.loads(match)
                # Validate format
                if all(isinstance(p, list) and len(p) == 2 for p in points):
                    return [[float(p[0]), float(p[1])] for p in points]

        # If no pattern matches, try to extract numbers and pair them
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if len(numbers) >= 2 and len(numbers) % 2 == 0:
            points = []
            for i in range(0, len(numbers), 2):
                points.append([float(numbers[i]), float(numbers[i + 1])])
            return points
        elif len(numbers) >= 2 and len(numbers) % 2 == 1:
            points = []
            for i in range(0, len(numbers)-1, 2):
                points.append([float(numbers[i]), float(numbers[i + 1])])
            return points

    except (json.JSONDecodeError, ValueError, IndexError) as e:
        print(f"Failed to parse trace: {e}")
        print(f"Text: {text}")
        return []


class ApiModel():

    def __init__(
        self,
        model_name: str,
        base_url:str,
        system_prompt: str,
        normalized_coordinates: bool = True,
        request_delay: float = 0.5,
        retry_delay: float = 2,
        max_retries: int = 3,
        max_tokens: int = 5000,
        temperature: float = 1.0,
    ):

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.normalized_coordinates = normalized_coordinates
        self.request_delay = request_delay
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Configure vLLM offline inference
        print(f"Loading processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Processor loaded.")
        
        print(f"Initializing vLLM model from {model_name}...")
        self.llm = LLM(
            model=model_name,
            mm_encoder_tp_mode="data",
            enable_expert_parallel=False,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        print("vLLM model initialized.")
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )


    def process_sample(self, sample: Dict, embodiment: str) -> Dict[str, Any]:

        # Extract fields
        sample_id = sample["sample_id"]
        category = sample["category"]
        image = sample["image"]
        task = sample["task"]

        # Format prompt (maintaining the same prompting format as before)
        prompt = self.system_prompt.format(embodiment=embodiment, task_str=task)

        # Convert image to RGB if necessary (for vLLM compatibility)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        # Prepare message content for vLLM (same structure, but using PIL Image directly)
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # Use PIL Image directly instead of base64
                },
                {"type": "text", "text": prompt},
            ]
        }]

        # Limit request rate
        time.sleep(self.request_delay)

        # Make vLLM inference with retries
        for attempt in range(self.max_retries):
            try:
                # Prepare inputs for vLLM
                vllm_inputs = prepare_inputs_for_vllm(messages, self.processor)
                
                # Generate using vLLM
                outputs = self.llm.generate([vllm_inputs], sampling_params=self.sampling_params)
                
                # Extract the response text
                response_text = outputs[0].outputs[0].text
                predicted_trace = parse_trace(response_text)
                
                # Unnormalize coordinates
                #assert self.normalized_coordinates, "Normalized coordinates are not supported"
                if self.normalized_coordinates:
                    width, height = image.size
                    predicted_trace = [
                        [int(x * width), int(y * height)] for x, y in predicted_trace
                    ]
                    print(f"Predicted trace: {predicted_trace}")

                # Extract reasoning if available (vLLM doesn't typically provide separate reasoning)
                reasoning_text = ""

                return {
                    "sample_id": sample_id,
                    "embodiment": embodiment,
                    "category": category,
                    "raw_response": response_text,
                    "reasoning": reasoning_text,
                    "prediction": predicted_trace,
                }

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return {
                        "sample_id": sample_id,
                        "embodiment": embodiment,
                        "category": category,
                        "raw_response": "",
                        "reasoning": "",
                        "prediction": [],
                    }

    def process_samples_batch(self, samples: List[Dict], embodiments_list: List[str]) -> List[Dict[str, Any]]:
        """
        Process all samples in batch for faster inference using vLLM.
        
        Args:
            samples: List of sample dictionaries
            embodiments_list: List of embodiments corresponding to each sample
            
        Returns:
            List of result dictionaries
        """
        # Prepare all inputs in batch
        print("Preparing all inputs for batch processing...")
        vllm_inputs = []
        metadata = []  # Store metadata (sample_id, embodiment, category) for each input
        
        for sample, embodiment in zip(samples, embodiments_list):
            # Extract fields
            sample_id = sample["sample_id"]
            category = sample["category"]
            image = sample["image"]
            task = sample["task"]
            
            # Format prompt (maintaining the same prompting format as before)
            prompt = self.system_prompt.format(embodiment=embodiment, task_str=task)
            
            # Convert image to RGB if necessary (for vLLM compatibility)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            # Prepare message content for vLLM
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # Use PIL Image directly instead of base64
                    },
                    {"type": "text", "text": prompt},
                ]
            }]
            
            # Prepare inputs for vLLM
            vllm_input = prepare_inputs_for_vllm(messages, self.processor)
            vllm_inputs.append(vllm_input)
            metadata.append({
                "sample_id": sample_id,
                "embodiment": embodiment,
                "category": category,
            })
        
        print(f"Prepared {len(vllm_inputs)} inputs for batch inference.")
        
        # Generate all samples in batch with retries
        for attempt in range(self.max_retries):
            try:
                #if True:
                print(f"Running batch inference (attempt {attempt + 1}/{self.max_retries})...")
                outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
                print("Batch inference completed successfully.")
                
                # Process all outputs
                results = []
                for i, output in enumerate(outputs):
                    # Extract the response text
                    response_text = output.outputs[0].text
                    
                    predicted_trace = parse_trace(response_text)
                    print(f"Response text: {response_text}")
                    print(f"Predicted trace: {predicted_trace}")
                    if self.normalized_coordinates:
                        width, height = samples[i]["image"].size  
                        predicted_trace = [
                            [int(x * width), int(y * height)] for x, y in predicted_trace
                        ]
                        #print(f"Predicted trace: {predicted_trace}")
                    
                    # Extract reasoning if available (vLLM doesn't typically provide separate reasoning)
                    reasoning_text = ""
                    
                    results.append({
                        "sample_id": metadata[i]["sample_id"],
                        "embodiment": metadata[i]["embodiment"],
                        "category": metadata[i]["category"],
                        "raw_response": response_text,
                        "reasoning": reasoning_text,
                        "prediction": predicted_trace,
                    })
                
                return results

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error: {e}")
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    # Return empty results for failed batch
                    return [{
                        "sample_id": meta["sample_id"],
                        "embodiment": meta["embodiment"],
                        "category": meta["category"],
                        "raw_response": "",
                        "reasoning": "",
                        "prediction": [],
                    } for meta in metadata]


# ### Validation Split

# In[ ]:

if __name__ == "__main__":
    VALIDATION = True
    if VALIDATION:
        # Create Model
        model = ApiModel(model_name, base_url, system_prompt)

        # Collect all samples and embodiments for batch processing
        print("Collecting all samples and embodiments...")
        samples_list = []
        embodiments_list = []
        dataset = dataset["validation"]
        
        for i, sample in tqdm(enumerate(dataset), desc="Collecting samples", total=len(dataset)):
            # Iterate over embodiments of a sample
            embodiments = sample["embodiments"]
            for embodiment in embodiments:
                samples_list.append(sample)
                embodiments_list.append(embodiment)
        
        print(f"Collected {len(samples_list)} samples for batch processing.")
        
        # Process all samples in batch
        print("Starting batch inference...")
        results = model.process_samples_batch(samples_list, embodiments_list)
        
        results_df = pd.DataFrame(results)

        # Save results

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"./tsv_logs/val_eval/{model_safe_name}_validation_{timestamp}.tsv"
        results_df.to_csv(
            results_path,
            sep="\t",
            index=False,
            encoding="utf-8",
        )

    else:
        # Create Model
        model = ApiModel(model_name, base_url, system_prompt)

        # Collect all samples and embodiments for batch processing
        print("Collecting all samples and embodiments...")
        samples_list = []
        embodiments_list = []
        dataset = dataset["test"]
        
        for i, sample in tqdm(enumerate(dataset), desc="Collecting samples", total=len(dataset)):
            # Iterate over embodiments of a sample
            embodiments = sample["embodiments"]
            for embodiment in embodiments:
                samples_list.append(sample)
                embodiments_list.append(embodiment)
        
        print(f"Collected {len(samples_list)} samples for batch processing.")
        
        # Process all samples in batch
        print("Starting batch inference...")
        results = model.process_samples_batch(samples_list, embodiments_list)
        
        results_df = pd.DataFrame(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"./tsv_logs/{model_safe_name}_test_{timestamp}.tsv"
        results_df.to_csv(
            results_path,
            sep="\t",
            index=False,
            encoding="utf-8",
        )

