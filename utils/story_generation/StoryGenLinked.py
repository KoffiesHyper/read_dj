import os
import re
import subprocess
import random
from typing import List, Tuple, Dict

from decouple import config

SG_PATH = config("SG_PATH")
LLAMA_PATH = config("LLAMA_PATH")

# === CONFIGURATION ===
LLAMA_CLI_PATH = f"{SG_PATH}/llama.cpp/build/bin/llama-cli"
MODEL_PATH = LLAMA_PATH
MAX_CONTEXT_TOKENS = 8192
MAX_GENERATION_TOKENS_PER_BATCH = 500


# === PARSING FUNCTIONS ===
def detect_story_number(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        print(filepath)
        content = f.read()
        print(content)
    match = re.search(r"--- Story Outline #(\d+) ---", content)
    if match:
        return int(match.group(1))    
    raise ValueError("Could not find a story number in the source file.")


def extract_outline_and_initial_paragraph(filepath: str, outline_number: int) -> Tuple[str, str, List[str], str]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    outline_number = detect_story_number(filepath)
    pattern = fr"--- Story Outline #{outline_number} ---\n(.*?)(?=(--- Story Outline #|\Z))"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find outline #{outline_number} in the file.")

    block = match.group(1)

    premise_match = re.search(r"Premise.*?:\s*(.*)", block)
    theme_match = re.search(r"Theme:\s*(.*)", block)
    outline_match = re.search(r"Outline:\s*(.*)", block)

    premise = premise_match.group(1).strip() if premise_match else ""
    theme = theme_match.group(1).strip() if theme_match else ""
    outline_line = outline_match.group(1).strip() if outline_match else ""
    outline = [e.strip() for e in outline_line.split(",")]

    first_event = outline[0]

    para_match = re.search(
        fr"--- Paragraph Output for Event: '{re.escape(first_event)}' ---\n([\s\S]*?)(?=^---|\Z)",
        block,
        re.MULTILINE
    )
    initial_paragraph = para_match.group(1).strip() if para_match else ""

    return premise, theme, outline, initial_paragraph


def extract_story_paragraphs_and_transitions(filepath: str, story_number: int) -> Tuple[List[str], List[str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    story_header = f"--- Story {story_number} ---"
    match = re.search(re.escape(story_header), text)
    if not match:
        raise ValueError(f"Story {story_number} not found in file.")

    story_start = match.end()
    rest = text[story_start:]

    block = re.split(r"--- Story \d+ ---", rest)[0]

    paragraphs = re.findall(r"\[Paragraph \d+\](.*?)\n(?=\[|\Z)", block, re.DOTALL)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    transitions = []
    if "--- Paragraph Transitions ---" in block:
        trans_block = block.split("--- Paragraph Transitions ---", 1)[-1]
        raw_trans = re.findall(r"\[Transition \d+\]\s*(.*?)(?=\n\[Transition|\Z)", trans_block, re.DOTALL)
        transitions = [t.strip() for t in raw_trans if t.strip()]

    return paragraphs, transitions


def load_event_mappings(filepath: str) -> Dict[str, str]:
    mappings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if ": " in line:
                key, val = line.split(": ", 1)
                clean_key = key.strip()
                if clean_key.startswith('"') and clean_key.endswith('"'):
                    clean_key = clean_key[1:-1]

                clean_val = val.strip()
                if clean_val.startswith('"') and clean_val.endswith('"'):
                    clean_val = clean_val[1:-1]

                mappings[clean_key] = clean_val
    return mappings

def load_mistakes(phoneme_file: str) -> list[str]:
    phonemes = []
    with open(phoneme_file, "r", encoding="utf-8") as f:
        for line in f:
            phonemes = line.split(" ")
    return phonemes


# === GENERATION CORE ===
def generate_continuation_paragraph(
    theme: str,
    outline_events: List[str],
    story_so_far: List[str],
    previous_source_paragraph: str,
    current_source_paragraph: str,
    previous_generated_paragraph: str,
    transition_text: str,
    mapped_event: str,
    phonemes: List[str],
    llama_cli_path: str,
    model_path: str,
    max_context_tokens: int,
    max_generation_tokens_per_batch: int
) -> str:
    prompt_body = f"""
    I want to generate the continuation paragraph of a children's story for age 6.
    
    Continue the story as follows:
        The story must include the following:
            Theme of the story: {theme}
            Event to cover: {mapped_event}
            Transition: {transition_text}
        
        It must use as reference the following:
            Previous paragraph from the original story (reference):
            {previous_source_paragraph}
            
            Current paragraph from the original story (reference):
            {current_source_paragraph}
            
        It must carry on from this paragraph:
            Previous paragraph generated:
            {previous_generated_paragraph}

        Using the above:
        - Write a new paragraph continuing the story.
        - Match the tone and theme.
        - Keep it imaginative and age-appropriate age 6.
        - Do not repeat prior paragraphs.
        
    Please incorporate words with the sounds of {phonemes[0]}, {phonemes[1]}, {phonemes[2]}, {phonemes[3]}, {phonemes[4]} 
    where it makes sense to.
    Spell the words as they are spelt not how they would sound. This is important because these are phonemes you are incorporating.
    
    The grammar and vocabulary should match that of what a 6 year old could understand.
    
    If you are working on generating your 6th paragraph, the story needs to conclude in that paragraph.

    Next paragraph:
    ###BEGIN###"""

    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + prompt_body
        + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    command = [
        llama_cli_path,
        "-m", model_path,
        "-c", str(max_context_tokens),
        "-n", str(max_generation_tokens_per_batch),
        "--temp", "0.7",
        "--top-p", "0.9",
        "--seed", str(random.randint(1, 10000)),
        "--prompt", prompt
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    output = result.stdout.strip()

    # Remove role markers if accidentally returned
    output = re.sub(r"<\|start_header_id\|>.*?<\|end_header_id\|>", "", output)
    output = output.replace("assistant", "").strip()

    if "###BEGIN###" in output:
        generated_text = output.split("###BEGIN###", 1)[-1].strip()
    else:
        generated_text = output  # fallback

    return generated_text


def build_story(
    outline: List[str],
    theme: str,
    initial_paragraph: str,
    source_story_paragraphs: List[str],
    transitions: List[str],
    event_mappings: Dict[str, str],
    phonemes: List[str]
) -> List[str]:
    story = []

    # Add initial paragraph with event label
    first_event = outline[0]
    story.append(f"--- Event: {first_event} ---\n{initial_paragraph}")

    for i in range(1, len(outline)):
        raw_event = outline[i]

        mapped_event = event_mappings.get(raw_event, f"a moment related to {raw_event.replace('_', ' ')}")
        transition = transitions[i - 1] if i - 1 < len(transitions) and transitions[i-1] != "No transition." else "Continue the story."

        previous_source = source_story_paragraphs[i - 1] if i - 1 < len(source_story_paragraphs) else ""
        current_source = source_story_paragraphs[i] if i < len(source_story_paragraphs) else ""

        # Remove event and transition labels from story_so_far benext_fore passing to generator
        cleaned_story = [re.sub(r'^--- .*? ---\n', '', p) for p in story]
        previous_generated = cleaned_story[-1] if cleaned_story else ""

        print(f"Generating paragraph {i+1} for event '{raw_event}' with transition: '{transition}'")
        new_paragraph = generate_continuation_paragraph(
            theme=theme,
            phonemes = load_mistakes(f"{SG_PATH}/phonemes.txt"),
            outline_events=outline,
            story_so_far=cleaned_story,
            previous_source_paragraph=previous_source,
            current_source_paragraph=current_source,
            previous_generated_paragraph=previous_generated,
            transition_text=transition,
            mapped_event=mapped_event,
            llama_cli_path=LLAMA_CLI_PATH,
            model_path=MODEL_PATH,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            max_generation_tokens_per_batch=MAX_GENERATION_TOKENS_PER_BATCH
        )

        formatted = f"--- Theme: {theme} ---\n--- Event: {raw_event} ---\n--- Transition: {transition} ---\n{new_paragraph}"
        story.append(formatted)

    return story, phonemes


# === MAIN ===
def run_story_gen():
    outline_file = f"{SG_PATH}/generated_story_outline.txt"
    source_file = f"{SG_PATH}/selected_story.txt"
    mapping_file = f"{SG_PATH}/event_mappings_AFP150-400.txt"
    output_dir = f"{SG_PATH}/generated_stories"
    

    os.makedirs(output_dir, exist_ok=True)

    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by each matched story block
    story_blocks = re.split(r"(?=--- Matched Story \d+)", content)

    event_mappings = load_event_mappings(mapping_file)
    idx = 0
    for block in story_blocks:
        if not block.strip():
            continue
        idx = idx + 1
        # Extract story number
        story_num_match = re.search(r"--- Matched Story (\d+)", block)
        if not story_num_match:
            print("Skipping block: no matched story number found.")
            continue
        story_number = int(story_num_match.group(1))

        # Extract paragraphs
        source_paragraphs = re.findall(r"\[Paragraph \d+\](.*?)\n(?=\[|\Z)", block, re.DOTALL)
        source_paragraphs = [p.strip() for p in source_paragraphs if p.strip()]

        # Extract transitions
        transitions = []
        if "--- Paragraph Transitions ---" in block:
            trans_block = block.split("--- Paragraph Transitions ---", 1)[-1]
            transitions = re.findall(r"\[Transition \d+\]\s*(.*?)(?=\n\[Transition|\Z)", trans_block, re.DOTALL)
            transitions = [t.strip() for t in transitions if t.strip()]

        try:
            # Load outline info and initial paragraph
            premise, theme, outline, initial_paragraph = extract_outline_and_initial_paragraph(outline_file, idx)
            
            if not source_paragraphs:
                print(f"[Story {story_number}] No source paragraphs found. Skipping.")
                continue
            print(f"[Story {story_number}] Generating story with {len(outline)} events and {len(source_paragraphs)} source paragraphs...")

            full_story, phonemes = build_story(
                outline=outline,
                theme=theme,
                initial_paragraph=initial_paragraph,
                source_story_paragraphs=source_paragraphs,
                transitions=transitions,
                event_mappings=event_mappings,
                phonemes = load_mistakes(f"{SG_PATH}/phonemes.txt")
            ) 
            output_path = f"{SG_PATH}/all_generated_stories.txt"
            with open(output_path, "a", encoding="utf-8") as f_out:
                f_out.write(f"===== GENERATED STORY {idx} (from Matched Story {story_number}) =====\n\n")
                f_out.write(f"--- Phonemes Incorporated: {phonemes[0]}, {phonemes[1]}, {phonemes[2]}, {phonemes[3]}, {phonemes[4]} ---\n")
                f_out.write(f"--- Theme: {theme} ---\n")
                # f_out.write(f"--- Event: {outline[0]} ---")
                for para in full_story:
                    para = para.replace("[end of text]", "").strip()
                    f_out.write(para + "\n\n")
                f_out.write("\n\n")

            print(f"[Story {story_number}] Done. Output appended to {output_path}")


        except Exception as e:
            print(f"[Story {story_number}] Error: {e}")

