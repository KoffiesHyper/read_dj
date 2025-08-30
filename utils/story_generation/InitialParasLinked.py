import re
from datetime import datetime
import os
import subprocess
import random
import json
from .Mistakes import get_mistakes

from decouple import config

SG_PATH = config("SG_PATH")
LLAMA_PATH = config("LLAMA_PATH")

# --- llama.cpp CLI Configuration ---
LLAMA_CLI_PATH = f"{SG_PATH}/llama.cpp/build/bin/llama-cli"
MODEL_PATH = LLAMA_PATH

MAX_CONTEXT_TOKENS = 8192
MAX_GENERATION_TOKENS_PER_BATCH = 500

# --- Input/Output Configuration ---
OUTLINE_FILE_PATH = f"{SG_PATH}/selected_outlines_AFP150-400.txt"
THEME_FILE_PATH = f"{SG_PATH}/cleaned_themes_AFP150-400.txt"
EVENT_MAPPING_FILE = f"{SG_PATH}/event_mappings_AFP150-400.txt"
MISTAKES_FILE = f"{SG_PATH}/phonemes.txt"

# --- Helper Function for Token Counting ---
def simple_token_estimate(text):
    return len(text.split()) * 1.3

# --- File Loading Functions ---
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

def load_event_mappings(mapping_file: str) -> dict[str, str]:
    mappings = {}
    try:
        if not os.path.exists(mapping_file):
            print(f"Warning: Event mapping file '{mapping_file}' not found.")
            return mappings

        with open(mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    json_object_string = "{" + line + "}"
                    parsed_dict = json.loads(json_object_string)
                    mappings.update(parsed_dict)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON line: {line} (Error: {e})")
        return mappings
    except Exception as e:
        print(f"Error loading event mappings: {e}")
        return {}

def load_all_outlines(outline_file: str) -> list[list[str]]:
    all_outlines = []
    try:
        if not os.path.exists(outline_file):
            print(f"Error: Outline file '{outline_file}' not found.")
            return []

        with open(outline_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                events = [e.strip() for e in line.split(',') if e.strip()]
                if events:
                    all_outlines.append(events)
        return all_outlines
    except Exception as e:
        print(f"Error reading outline file: {e}")
        return []
    
def load_mistakes(phoneme_file: str) -> list[str]:
    phonemes = []
    with open(phoneme_file, "r", encoding="utf-8") as f:
        for line in f:
            phonemes = line.split(" ")
    return phonemes
            

# --- Obstacle Prompt Template ---
obstacle_prompt_template = """Given the story premise: {premise}
Write a list of five unique obstacles that could block the protagonist's major goal in the story.
Each obstacle should be a short phrase, less than 5 words.
Format as a numbered list.
"""

# --- Core Generation Function ---
def generate_paragraph_core(
    event_prompt: str,
    theme_prompt: str,
    llama_cli_path: str,
    model_path: str,
    max_context_tokens: int,
    max_generation_tokens_per_batch: int,
    raw_event_name: str,
    user_input_text: str
) -> str:

    instruction_prompt_template = """
    
    I am wanting to write the first paragraphs of some children's stories to teach reading.
    
    Your task is to generate a single paragraph, exactly 3 sentences long, as part of a children's 
    story for a 6 year old.
    
    The story should have the theme of '{theme}'.
    
    This paragraph must incorporate the event: {event}.
    
    Please incorporate words with the sounds of {phonemes[0]}, {phonemes[1]}, {phonemes[2]}, {phonemes[3]}, {phonemes[4]} 
    where it makes sense to. 
    Spell the words as they are spelt not how they would sound. This is important because these are phonemes you are incorporating.

    The grammar and vocabulary should match that of what a 6 year old could understand.
    
    Generate the paragraph now:
    """
    final_event_instruction = event_prompt

    if raw_event_name == "add_obstacle_towards_major_goal":
        obstacle_hint_prompt = obstacle_prompt_template.format(premise=user_input_text)
        print(f"Generating obstacle hints for premise: '{user_input_text}'")

        command_hint = [
            llama_cli_path,
            "-m", model_path,
            "-c", str(max_context_tokens),
            "-n", "100",
            "--temp", "0.7",
            "--top-p", "0.9",
            "--seed", str(random.randint(1, 10000)),
            "--prompt", "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        f"{obstacle_hint_prompt}"
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ]
        try:
            result = subprocess.run(command_hint, capture_output=True, text=True, check=True)
            raw_hint_output = result.stdout

            assistant_marker = "assistant\n\n"
            start = raw_hint_output.rfind(assistant_marker)
            clean_hint_output = raw_hint_output[start + len(assistant_marker):] if start != -1 else raw_hint_output
            clean_hint_output = clean_hint_output.replace("<|eot_id|>", "").replace("[end of text]", "").strip()

            hints = []
            for line in clean_hint_output.split('\n'):
                if line.strip() and re.match(r'^\d+\.', line):
                    hints.append(re.sub(r'^\d+\.\s*', '', line.strip()))

            joined_hints = ", ".join(hints[:5])
            print(f"Generated Obstacle Hints: {joined_hints}\n")

            if "{{obstacle_hint}}" in event_prompt:
                final_event_instruction = event_prompt.replace("{{obstacle_hint}}", f"possible examples include: {joined_hints}")
            else:
                final_event_instruction = f"{event_prompt}. Consider obstacles such as: {joined_hints}"

        except subprocess.CalledProcessError as e:
            print("ERROR generating obstacle hints:", e)
            final_event_instruction = event_prompt

    # Prepare final story generation prompt
    prompt = instruction_prompt_template.format(event=final_event_instruction, theme=theme_prompt, phonemes = load_mistakes(f"{SG_PATH}/phonemes.txt"))
    full_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    print("\n🟡 Final Prompt Sent to llama.cpp:\n", full_prompt, "\n---------------------------")

    command = [
        llama_cli_path,
        "-m", model_path,
        "-c", str(max_context_tokens),
        "-n", str(max_generation_tokens_per_batch),
        "--temp", "0.7",
        "--top-p", "0.9",
        "--seed", "1",
        "--prompt", full_prompt
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        raw_output = result.stdout
        print("\n🟢 Raw Output from llama-cli:\n", raw_output, "\n---------------------------")

        assistant_marker = "assistant\n\n"
        start = raw_output.rfind(assistant_marker)
        output_clean = raw_output[start + len(assistant_marker):] if start != -1 else raw_output
        output_clean = output_clean.replace("<|eot_id|>", "").replace("[end of text]", "").strip()

        # Improved paragraph extraction: find first paragraph that looks like story content
        paragraph_candidates = [p.strip() for p in output_clean.split('\n\n') if p.strip()]
        paragraph = ""

        for candidate in paragraph_candidates:
            if re.match(r"^[A-Z].*[.?!]$", candidate):
                paragraph = candidate
                break

        if not paragraph:
            paragraph = "[No valid paragraph found]"

        paragraph = paragraph.replace("[end of text]", "").strip()

        print("\n✅ Final Cleaned Paragraph:\n", paragraph if paragraph else "[EMPTY RESPONSE]", "\n---------------------------")
        return paragraph

    except subprocess.CalledProcessError as e:
        print("ERROR during paragraph generation:", e)
        return ""

# --- Process One Outline ---
def process_single_story_outline(outline_raw_events, story_index, story_premise, selected_theme, event_mappings,
                                 llama_cli_path, model_path, max_context_tokens, max_generation_tokens_per_batch):
    output_filename = f"{SG_PATH}/generated_story_outline.txt"
    print(f"\n--- Generating initial paragraph for Outline #{story_index + 1} to '{output_filename}' ---")
    print(f"Story Premise (from Theme Sentence): '{story_premise}'")
    print(f"Overall Theme: '{selected_theme}'")

    with open(output_filename, "a", encoding="utf-8") as f:
        f.write(f"--- Story Outline #{story_index + 1} ---\n")
        f.write(f"Premise (from Theme Sentence): {story_premise}\n")
        f.write(f"Theme: {selected_theme}\n")
        f.write(f"Outline: {outline_raw_events[0]},")
        f.write(f"{outline_raw_events[1]},")
        f.write(f"{outline_raw_events[2]},")
        f.write(f"{outline_raw_events[3]},")
        f.write(f"{outline_raw_events[4]},")
        f.write(f"{outline_raw_events[5]},")
        f.write(f"{outline_raw_events[6]}\n\n")

    if outline_raw_events:
        raw_event = outline_raw_events[0]
        mapped_event = event_mappings.get(raw_event, f"an event related to: {raw_event.replace('_', ' ')}")
        print(f"\nProcessing Initial Event ('{raw_event}'): {mapped_event}")

        paragraph = generate_paragraph_core(
            event_prompt=mapped_event,
            theme_prompt=selected_theme,
            llama_cli_path=llama_cli_path,
            model_path=model_path,
            max_context_tokens=max_context_tokens,
            max_generation_tokens_per_batch=max_generation_tokens_per_batch,
            raw_event_name=raw_event,
            user_input_text=story_premise
        )

        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"--- Paragraph Output for Event: '{raw_event}' ---\n")
            f.write(paragraph if paragraph else "[No output generated]")
            f.write("\n\n")
            
    else:
        print(f"Warning: Outline #{story_index + 1} has no events.")

# --- Main ---
def run_inital_paras(mistakes):
    get_mistakes(mistakes)
    themes = load_themes(THEME_FILE_PATH)
    if not themes:
        print("No themes loaded.")
        exit()

    event_mappings = load_event_mappings(EVENT_MAPPING_FILE)
    outlines = load_all_outlines(OUTLINE_FILE_PATH)

    num_stories = 1
    print(f"\nProcessing {num_stories} stories based on available outlines and theme descriptions.")

    # for i in range(num_stories):
    i = random.randint(0, 249)
    outline = outlines[i]
    theme_word, theme_sentence = themes[i % len(themes)]
    process_single_story_outline(
        outline_raw_events=outline,
        story_index=i,
        story_premise=theme_sentence,
        selected_theme=theme_word,
        event_mappings=event_mappings,
        llama_cli_path=LLAMA_CLI_PATH,
        model_path=MODEL_PATH,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_generation_tokens_per_batch=MAX_GENERATION_TOKENS_PER_BATCH
    )

    print("\nAll initial paragraphs processed. Check the 'generated_story_outline_X.txt' files.")
