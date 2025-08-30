import os
import re
import subprocess
import random
from typing import List, Tuple

from decouple import config

SG_PATH = config("SG_PATH")

# === CONFIGURATION ===
LLAMA_CLI_PATH = f"{SG_PATH}/llama.cpp/build/bin/llama-cli"
MODEL_PATH = f"{SG_PATH}/llama.cpp/hf_models/LLaMA-3.2-3B-Instruct-Q4_K_M.gguf"
MAX_CONTEXT_TOKENS = 8192
MAX_GENERATION_TOKENS_PER_BATCH = 500
THEME_FILE_PATH = f"{SG_PATH}/cleaned_themes_AFP150-400.txt"

# === PARSING FUNCTIONS ===
def detect_story_number(filepath: str) -> int:
    print(f"Detecting story number from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"--- Story (\d+) ---", content)
    if match:
        number = int(match.group(1))
        print(f"Found story number: {number}")
        return number
    raise ValueError("Could not find a story number in the source file.")

def extract_outline_and_initial_paragraph(filepath: str, outline_number: int) -> Tuple[str, List[str], str]:
    print(f"Extracting outline and initial paragraph for outline #{outline_number} from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    outline_number = detect_story_number(filepath)
    pattern = fr"--- Story Outline #{outline_number} ---\n(.*?)(?=(--- Story Outline #|\Z))"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find outline #{outline_number} in the file.")

    block = match.group(1)

    theme_match = re.search(r"Theme:\s*(.*)", block)
    outline_match = re.search(r"Outline:\s*(.*)", block)

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

    print(f"Extracted theme: {theme}")
    print(f"First outline event: {first_event}")
    print(f"Initial paragraph length: {len(initial_paragraph)} characters")

    return theme, outline, initial_paragraph

def extract_story_paragraphs_and_transitions(filepath: str, story_number: int) -> Tuple[List[str], List[str]]:
    print(f"Extracting story paragraphs and transitions for story {story_number} from {filepath}")
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

    print(f"Found {len(paragraphs)} source paragraphs")
    print(f"Found {len(transitions)} transitions")

    return paragraphs, transitions

def load_themes(theme_file: str) -> list[tuple[str, str]]:
    themes_data = []
    try:
        if not os.path.exists(theme_file):
            print(f"Error: Theme file '{theme_file}' not found.")
            return []

        with open(theme_file, "r", encoding="utf-8") as f:
            content = f.read()

        raw_blocks = re.split(r'---\s*\n', content)
        theme_word_pattern = re.compile(r"Theme Word:\s*(\w+)", re.IGNORECASE)
        theme_sentence_pattern = re.compile(r"Theme Sentence:\s*(.+)", re.IGNORECASE)

        for block_content in raw_blocks:
            if not block_content.strip():
                continue
            word_match = theme_word_pattern.search(block_content)
            sentence_match = theme_sentence_pattern.search(block_content)

            theme_word = word_match.group(1).strip() if word_match else None
            theme_sentence = sentence_match.group(1).strip() if sentence_match else None

            if theme_word and theme_sentence:
                themes_data.append((theme_word, theme_sentence))
            else:
                if block_content.strip():
                    print(f"Warning: Incomplete theme block found:\n{block_content.strip()}\nMissing 'Theme Word:' or 'Theme Sentence:'.")

        if not themes_data:
            print(f"Warning: No complete 'Theme Word:' and 'Theme Sentence:' pairs found in '{theme_file}' using the expected format.")
        return themes_data

    except Exception as e:
        print(f"Error reading theme file or parsing themes: {e}")
        return []
    
def load_mistakes(phoneme_file: str) -> list[str]:
    phonemes = []
    with open(phoneme_file, "r", encoding="utf-8") as f:
        for line in f:
            phonemes = line.split(" ")
    return phonemes

# === GENERATION CORE ===
def generate_continuation_paragraph(
    theme: str,
    story_so_far: List[str],
    previous_source_paragraph: str,
    current_source_paragraph: str,
    transition_text: str,
    llama_cli_path: str,
    model_path: str,
    max_context_tokens: int,
    max_generation_tokens_per_batch: int,
    phonemes: List[str]
) -> str:
    prompt_body = f"""
    
    I want to generate the continuation paragraph of a children's story for age 6.
    
    Continue the story as follows:
        The story must include the following:
            Theme: {theme}
            Transition: {transition_text}
            
        It must use as reference the following:
            Previous paragraph from the original story (reference):
            {previous_source_paragraph}
            
            Current paragraph from the original story (reference):
            {current_source_paragraph}
            
        It must carry on from this paragraph:
            Previous paragraph generated:
            {story_so_far[-1] if story_so_far else ""}

        Using all the above:
            - Write a new paragraph continuing the story.
            - Match the tone and theme.
            - Keep it imaginative and age-appropriate for age 6.
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

    print(f"Calling llama-cli with prompt size: {len(prompt)} characters")
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

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("Error: llama-cli subprocess timed out.")
        return "[ERROR: Generation timed out]"

    if result.returncode != 0:
        print("Error: llama-cli returned non-zero exit code.")
        print("stderr:", result.stderr)
        return "[ERROR: llama-cli failed]"

    if "###BEGIN###" in output:
        generated_text = output.split("###BEGIN###", 1)[-1].strip()
    else:
        generated_text = output.strip()

    print(f"Generated paragraph length: {len(generated_text)} characters")

    return generated_text

def build_story(
    theme: str,
    initial_paragraph: str,
    source_story_paragraphs: List[str],
    transitions: List[str],
    phonemes: List[str],
    target_paragraph_count: int = 7,
) -> List[str]:
    print("Building story...")
    story = [initial_paragraph]

    # Ensure enough paragraphs and transitions to reach target count + 1 (for next paragraph)
    while len(transitions) < target_paragraph_count - 1:
        transitions.append("Continue the story.")
    while len(source_story_paragraphs) < target_paragraph_count:
        source_story_paragraphs.append("")

    print(f"Target paragraph count: {target_paragraph_count}")
    print(f"Using {len(source_story_paragraphs)} source paragraphs")
    print(f"Using {len(transitions)} transitions")

    for i in range(1, target_paragraph_count):
        print(f"--- Generating paragraph {i + 1} ---")
        transition = transitions[i - 1]
        prev_source = source_story_paragraphs[i - 1] if i - 1 >= 0 else ""
        current_source = source_story_paragraphs[i]

        new_paragraph = generate_continuation_paragraph(
            theme=theme,
            phonemes = load_mistakes(f"{SG_PATH}/phonemes.txt"),
            story_so_far=story,
            previous_source_paragraph=prev_source,
            current_source_paragraph=current_source,
            transition_text=transition,
            llama_cli_path=LLAMA_CLI_PATH,
            model_path=MODEL_PATH,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            max_generation_tokens_per_batch=MAX_GENERATION_TOKENS_PER_BATCH
        )

        formatted = new_paragraph.strip().split("\n")[-1]
        story.append(formatted)

    return story, phonemes


def run_no_outline_gen():
    source_file = f"{SG_PATH}/selected_story.txt"
    output_file = f"{SG_PATH}/generated_story_output_no_outline.txt"

    # Clear output file at start
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write("")

    print("===== MULTI-STORY GENERATION (NO OUTLINE) START =====")

    with open(source_file, "r", encoding="utf-8") as f:
        all_text = f.read()

    # Split by each Matched Story block, keeping the header
    blocks = re.split(r"(?=--- Matched Story \d+ .*?---)", all_text)

    result = []

    idx = 0
    for block in blocks:
        if not block.strip():
            continue
        idx = idx + 1
        # Extract story number from block header
        story_id_match = re.search(r"--- Matched Story (\d+) .*?---", block)

        if not story_id_match:
            continue  # Skip if no header found

        story_id = int(story_id_match.group(1))
        print(f"\n=== Processing Matched Story {story_id} ===")

        # Extract paragraphs and transitions from this block text
        try:
            paragraphs, transitions = extract_story_paragraphs_and_transitions(
                filepath=None,  # We'll pass block text directly instead of file
                story_number=story_id
            )
        except Exception:
            # If your original function expects file input, we'll adjust:
            # So let's create a helper version for block text directly:
            def extract_story_paragraphs_and_transitions_from_text(text: str):
                paragraphs = re.findall(r"\[Paragraph \d+\](.*?)\n(?=\[|\Z)", text, re.DOTALL)
                paragraphs = [p.strip() for p in paragraphs if p.strip()]

                transitions = []
                if "--- Paragraph Transitions ---" in text:
                    trans_block = text.split("--- Paragraph Transitions ---", 1)[-1]
                    raw_trans = re.findall(r"\[Transition \d+\]\s*(.*?)(?=\n\[Transition|\Z)", trans_block, re.DOTALL)
                    transitions = [t.strip() for t in raw_trans if t.strip()]

                return paragraphs, transitions

            paragraphs, transitions = extract_story_paragraphs_and_transitions_from_text(block)

        if not paragraphs:
            print(f"❌ No paragraphs found for Matched Story {story_id}, skipping.")
            continue

        themes = load_themes(THEME_FILE_PATH)
        if not themes:
            print("No themes loaded.")
            exit()
        random_int = random.randint(0, len(themes))
        # Since no outlines, theme can be empty or fixed default
        # theme = "A fun and imaginative children's story"
        theme, theme_sentence = themes[random_int]

        # Use first paragraph as initial paragraph
        initial_paragraph = paragraphs[0]

        # Build story continuations (target 7 paragraphs including initial)
        full_story, phonemes = build_story(theme, initial_paragraph, paragraphs, transitions, phonemes = load_mistakes(f"{SG_PATH}/phonemes.txt"), target_paragraph_count=7)

        result = []

        print(full_story)

        # Write story block output to file
        with open(output_file, "a", encoding="utf-8") as f_out:
            f_out.write(f"--- Generated Story {idx} (From Matched Story {story_id}) ---\n\n")
            f_out.write(f"--- Phonemes Incorporated: {phonemes[0]}, {phonemes[1]}, {phonemes[2]}, {phonemes[3]}, {phonemes[4]} ---\n")
            f_out.write(f"--- Theme: {theme} ---\n")
            for para in full_story:
                clean_para = para.replace("[end of text]", "").replace("assistant", "").strip()
                result.append(clean_para)
                f_out.write(clean_para + "\n\n")

        print(f"✅ Finished Matched Story {story_id} and appended to output.")

    print("===== MULTI-STORY GENERATION COMPLETE =====")
    return result